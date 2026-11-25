#include <assert.h>
#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Number of possible byte values.
#define MAX_SYMBOL_COUNT 256
// Exact node count for k >= 1 leaves is 2k - 1: each merge turns two free
// nodes into one, so k leaves need k - 1 merges, i.e. k - 1 internal nodes.
#define MAX_TREE_NODES (2 * MAX_SYMBOL_COUNT - 1)
// 2^40 bytes = 1 TiB. Limits the maximum code length: a tree of depth d
// needs at least F(d + 3) - 1 input bytes (Fibonacci frequencies give the
// deepest tree), so for n <= 2^40 no code is longer than 56 bits.
#define MAX_INPUT_SIZE ((size_t)1 << 40)
// Maximum code length guaranteed by MAX_INPUT_SIZE. Codes are stored in
// uint64_t.
#define MAX_CODE_LENGTH 56

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// bit stream implementation

typedef struct {
	uint8_t *data;
	size_t bit_pos;
} bit_stream;

// Creates a bit stream over the given buffer, positioned at bit 0.
// The buffer is not copied: the caller owns it and must keep it alive
// while the stream is in use. Bits are read and written MSB-first.
bit_stream bit_stream_new(uint8_t *data) {
	return (bit_stream){data, 0};
}

// Prints the first n bytes of data in binary, MSB first, separated by spaces.
// Debug helper.
void print_n_bytes(const uint8_t *data, size_t n) {
	for (size_t i = 0; i < n; i++) {
		for (int bit = 7; bit >= 0; bit--) {
			putchar((data[i] >> bit) & 1 ? '1' : '0');
		}
		putchar(' ');
	}
	putchar('\n');
}

// Writes the lowest value_length bits of value into the stream, MSB first.
// Bits above value_length are ignored. Does not check bounds: the buffer
// must be large enough and zero-initialized (bits are set with |=).
void bit_stream_write(bit_stream *stream, uint64_t value, uint8_t value_length) {
	while (value_length != 0) {
		// Number of bits already filled in the current byte.
		uint8_t shift = stream->bit_pos % 8;
		size_t byte_index = stream->bit_pos / 8;

		// Number of bits to write in this step: as many as fit in the current byte.
		uint8_t to_write = MIN(8 - shift, value_length);
		// to_write ones in the lowest bits.
		uint64_t mask = ((uint64_t)1 << to_write) - 1;
		// Move the next to_write bits (the highest of the remaining ones) to the lowest positions.
		uint64_t chunk = value >> (value_length - to_write);
		// Drop higher bits that were already written in previous steps.
		chunk &= mask;
		// Place the bits right after the already filled bits of the current byte.
		chunk <<= (8 - shift - to_write);
		// chunk fits into 8 bits here, so the cast does not lose data.
		stream->data[byte_index] |= (uint8_t)chunk;

		value_length -= to_write;
		stream->bit_pos += to_write;
	}
}

// Reads the next bit from the stream (MSB first) and advances the position.
// Returns 0 or 1. Does not check bounds: the caller must ensure that
// bit_pos < buffer_size * 8.
uint8_t bit_stream_read_next(bit_stream *stream) {
	// Bit position in the current byte, counting from the MSB.
	uint8_t bit_shift = stream->bit_pos % 8;
	size_t byte_index = stream->bit_pos / 8;
	stream->bit_pos++;
	// Move the target bit to the lowest position and drop the rest.
	return (stream->data[byte_index] >> (7 - bit_shift)) & 1;
}

// huffman encoding/decoding implementation

// Node of a Huffman tree.
typedef struct {
	uint16_t parent;
	size_t frequency;
} huffman_node;

// Helper to debug nodes.
void print_node(uint16_t index, huffman_node *nodes) {
	huffman_node node = nodes[index];

	if (node.frequency == 0) {
		return;
	}

	printf("symbol: '%c', index: %d, frequency: %zu\n", index, index, node.frequency);
}

// Finds the free node (one without a parent) with the lowest frequency.
// ignore_index excludes one node from the search, so the second call
// does not return the node found by the first one (pass -1 to ignore none).
// On equal frequencies the node with the lowest index wins.
// Returns -1 if there is no such node.
int find_least_frequency_node_index(const huffman_node *nodes, int ignore_index) {
	size_t min_freq = SIZE_MAX;
	int min_frequency_index = -1;

	for (int i = 0; i < MAX_TREE_NODES; i++) {
		const huffman_node *node = &nodes[i];

		// Skip empty slots: symbols absent from the input and
		// internal nodes not created yet. A real node has frequency >= 1.
		if (node->frequency == 0) {
			continue;
		}

		if (i == ignore_index) {
			continue;
		}

		// Skip nodes that are already merged (have a parent).
		// A parent is always an internal node with index >= MAX_SYMBOL_COUNT,
		// so 0 means "no parent": a leaf can never be a parent.
		if (node->parent >= MAX_SYMBOL_COUNT) {
			continue;
		}

		if (node->frequency < min_freq) {
			min_freq = node->frequency;
			min_frequency_index = i;
		}
	}

	return min_frequency_index;
}

// Builds canonical Huffman codes from code lengths.
//
// Symbols are ordered by code length, then by symbol value. The first symbol
// gets code 0, each next symbol of the same length gets the previous code + 1,
// and moving to the next length appends a 0 bit (shift left by 1).
//
// Outputs:
//   canonical_codes[MAX_SYMBOL_COUNT]          - code of each symbol (encoder side);
//   first_code_by_length[MAX_CODE_LENGTH + 1]  - first code of each length;
//   count_codes_by_length[MAX_CODE_LENGTH + 1] - number of codes of each length;
//   symbols[MAX_SYMBOL_COUNT]                  - symbols in canonical order.
// Arrays indexed by length use indices 1..MAX_CODE_LENGTH; index 0 is unused.
// count_codes_by_length must be zero-initialized by the caller.
//
// Returns:
//   0 - ok (also for zero or one symbol, where the completeness check does not apply);
//   1 - lengths do not form a complete prefix code;
//   2 - a code is longer than MAX_CODE_LENGTH.
int build_canonical_codes(uint8_t const code_length[MAX_SYMBOL_COUNT],
						  uint64_t *canonical_codes,
						  uint64_t *first_code_by_length,
						  uint16_t *count_codes_by_length,
						  uint8_t *symbols) {
	size_t max_code_length = 0;
	for (size_t i = 0; i < MAX_SYMBOL_COUNT; i++) {
		max_code_length = MAX(max_code_length, code_length[i]);
	}

	// Must be checked before the loop: lengths may come from untrusted input,
	// and longer codes would not fit into uint64_t (shifts by >= 64 are UB).
	if (max_code_length > MAX_CODE_LENGTH) {
		return 2;
	}

	uint64_t code = 0;
	size_t unique_count = 0;
	// Iterating lengths in the outer loop and symbols in the inner loop
	// visits symbols in canonical order without sorting.
	for (size_t current_code_length = 1; current_code_length <= max_code_length; current_code_length++) {
		first_code_by_length[current_code_length] = code;

		for (size_t symbol = 0; symbol < MAX_SYMBOL_COUNT; symbol++) {
			if (code_length[symbol] == current_code_length) {
				symbols[unique_count] = (uint8_t)symbol;
				unique_count++;
				count_codes_by_length[current_code_length]++;
				canonical_codes[symbol] = code;
				code++;
			}
		}

		// Shift on every length, even if it has no symbols.
		code <<= 1;
	}

	// Zero or one symbol: nothing to check. A single symbol gets an artificial
	// length of 1 and covers only one of two slots, so the tree is not complete.
	if (unique_count <= 1) {
		return 0;
	}

	// Kraft check: a leaf of length L covers 2^(max - L) slots on the deepest
	// level, and a complete tree covers all 2^max of them. code counts the
	// covered slots, so after the last length it equals 2^max, and after the
	// final shift - 2^(max + 1).
	if (code == ((uint64_t)1 << (max_code_length + 1))) {
		return 0;
	}

	return 1;
}

typedef enum {
	HUFFMAN_INPUT_TOO_LARGE = 1,
	HUFFMAN_MALLOC_FAILED = 2,
	HUFFMAN_INVALID_CODE_LENGTH = 3,
	HUFFMAN_INVALID_DATA = 4
} huffman_error_code;

typedef struct {
	uint8_t *encoded;
	size_t encoded_size;
	size_t original_size;
	uint8_t code_length[MAX_SYMBOL_COUNT];
	huffman_error_code error_code;
} huffman_encoded;

// Encodes data of length n with a canonical Huffman code.
huffman_encoded huffman_encode(const uint8_t *data, size_t n) {
	huffman_encoded result = {NULL, 0, n, {0}, 0};

	if (n > MAX_INPUT_SIZE) {
		result.error_code = HUFFMAN_INPUT_TOO_LARGE;
		return result;
	}

	if (n == 0) {
		return result;
	}

	// Count symbol frequencies. Leaves live at indices 0..255 (index = symbol),
	// frequency == 0 means the symbol is absent.
	huffman_node nodes[MAX_TREE_NODES] = {0};
	uint16_t unique_count = 0;
	for (size_t i = 0; i < n; i++) {
		uint8_t symbol = data[i];
		huffman_node *node = nodes + symbol;

		if (node->frequency == 0) {
			unique_count++;
		}

		node->frequency++;
	}

	// Build the tree: repeatedly merge two free nodes with the lowest
	// frequencies. Each merge reduces the number of free nodes by one,
	// the last remaining node is the root. Internal nodes are appended
	// starting from index MAX_SYMBOL_COUNT.
	uint16_t merge_count = 0;
	for (size_t free_nodes = unique_count; free_nodes > 1; free_nodes--) {
		int left_idx = find_least_frequency_node_index(nodes, -1);
		assert(left_idx != -1);

		int right_idx = find_least_frequency_node_index(nodes, left_idx);
		assert(right_idx != -1);

		uint16_t new_node_idx = MAX_SYMBOL_COUNT + merge_count++;

		nodes[left_idx].parent = new_node_idx;
		nodes[right_idx].parent = new_node_idx;

		huffman_node new_node = {0, nodes[left_idx].frequency + nodes[right_idx].frequency};
		nodes[new_node_idx] = new_node;
	}

	// Code length of a symbol is the depth of its leaf: walk up via parent
	// until the root (the only node with parent == 0).
	// Also count the size of the encoded data in bits.
	size_t encoded_size_bits = 0;
	for (size_t i = 0; i < MAX_SYMBOL_COUNT; i++) {
		huffman_node *node = nodes + i;

		if (node->frequency == 0) {
			continue;
		}

		uint8_t code_length = 0;
		do {
			code_length++;
			node = nodes + node->parent;
		} while (node->parent != 0);

		result.code_length[i] = code_length;
		encoded_size_bits += code_length * nodes[i].frequency;
	}
	result.encoded_size = (encoded_size_bits + 7) / 8;

	// Lengths come from our own tree, so an error here is a bug, not bad input.
	uint64_t canonical_codes[MAX_SYMBOL_COUNT] = {0};
	uint64_t first_code_by_length[MAX_CODE_LENGTH + 1] = {0};
	uint16_t count_codes_by_length[MAX_CODE_LENGTH + 1] = {0};
	uint8_t symbols[MAX_SYMBOL_COUNT] = {0};
	if (build_canonical_codes(result.code_length, canonical_codes, first_code_by_length, count_codes_by_length, symbols)) {
		assert(0 && "invalid huffman tree");
	}

	// Encode message.
	result.encoded = calloc(result.encoded_size, 1);
	if (!result.encoded) {
		result.error_code = HUFFMAN_MALLOC_FAILED;
		result.encoded_size = 0;
		return result;
	}
	bit_stream stream = bit_stream_new(result.encoded);
	for (size_t i = 0; i < n; i++) {
		uint8_t symbol = data[i];
		bit_stream_write(&stream, canonical_codes[symbol], result.code_length[symbol]);
	}

	return result;
}

typedef struct {
	uint8_t *decoded;
	size_t decoded_size;
	huffman_error_code error_code;
} huffman_decoded;

// Decodes data produced by huffman_encode.
//
// encoded       - encoded bytes;
// encoded_size  - number of encoded bytes;
// code_lengths  - code length of each of MAX_SYMBOL_COUNT symbols;
// original_size - number of symbols to decode. Needed to stop before the
//                 padding bits in the last byte, which may look like valid codes.
//
// The input is treated as untrusted: invalid code lengths or corrupted data
// produce an error instead of out-of-bounds access.
//
// On success the caller owns result.decoded and must free() it.
// Empty output is not an error: returns decoded = NULL, decoded_size = 0.
// On error decoded = NULL, decoded_size = 0 and result.error_code is set.
huffman_decoded huffman_decode(uint8_t *encoded, size_t encoded_size, uint8_t const *code_lengths, size_t original_size) {
	huffman_decoded result = {NULL, original_size, 0};

	if (original_size == 0) {
		return result;
	}

	// Rebuild the same canonical codes as the encoder did.
	// canonical_codes is not needed for decoding, but the function fills it anyway.
	uint64_t canonical_codes[MAX_SYMBOL_COUNT];
	uint64_t first_code_by_length[MAX_CODE_LENGTH + 1] = {0};
	uint16_t count_codes_by_length[MAX_CODE_LENGTH + 1] = {0};
	uint8_t symbols[MAX_SYMBOL_COUNT] = {0};
	if (build_canonical_codes(code_lengths, canonical_codes, first_code_by_length, count_codes_by_length, symbols) != 0) {
		result.decoded_size = 0;
		result.error_code = HUFFMAN_INVALID_CODE_LENGTH;
		return result;
	}

	// symbol_offsets[L] - index in symbols where symbols of length L start,
	// i.e. the number of symbols with shorter codes.
	uint16_t symbol_offsets[MAX_CODE_LENGTH + 1] = {0};
	for (size_t len = 1; len <= MAX_CODE_LENGTH; len++) {
		symbol_offsets[len] = symbol_offsets[len - 1] + count_codes_by_length[len - 1];
	}

	result.decoded = malloc(original_size);
	if (!result.decoded) {
		result.decoded_size = 0;
		result.error_code = HUFFMAN_MALLOC_FAILED;
		return result;
	}

	// Read bits one by one, accumulating code and its length. In a canonical
	// code all codes of the same length are consecutive numbers, so code of
	// length L is valid if first[L] <= code < first[L] + count[L], and its
	// symbol is symbols[offset[L] + (code - first[L])].
	bit_stream stream = bit_stream_new(encoded);
	uint64_t code = 0;
	uint8_t code_length = 0;
	size_t bytes_written = 0;
	// The loop bound also keeps bit_stream_read_next within the buffer.
	for (size_t i = 0; i < encoded_size * 8; i++) {
		code = (code << 1) | bit_stream_read_next(&stream);
		code_length++;

		// No code matched up to the maximum length: the data is corrupted.
		// Possible even with valid lengths, e.g. a 1 bit when there is
		// a single symbol with code 0. Must be checked before using
		// code_length as an index.
		if (code_length > MAX_CODE_LENGTH) {
			free(result.decoded);
			result.decoded = NULL;
			result.decoded_size = 0;
			result.error_code = HUFFMAN_INVALID_DATA;
			return result;
		}

		// Half-open range [first, end): with "first + count - 1" an empty
		// length with first == 0 would give 0 - 1 = UINT64_MAX and match any code.
		uint64_t first = first_code_by_length[code_length];
		uint64_t end = first + count_codes_by_length[code_length];

		if (code >= first && code < end) {
			result.decoded[bytes_written] = symbols[symbol_offsets[code_length] + (code - first)];
			bytes_written++;
			// Stop here: the remaining bits are padding.
			if (bytes_written == original_size) {
				break;
			}

			code = 0;
			code_length = 0;
		}
	}

	// Bits ran out before all symbols were decoded: the data is truncated or corrupted.
	if (bytes_written != original_size) {
		free(result.decoded);
		result.decoded = NULL;
		result.decoded_size = 0;
		result.error_code = HUFFMAN_INVALID_DATA;
		return result;
	}

	return result;
}

// Simple example of file encoding and decoding with remaining functionality.
int main(void) {
	FILE *f_main = fopen("main", "rb");
	if (!f_main) {
		fprintf(stderr, "failed to open file: %s\n", strerror(errno));
		return 1;
	}

	fseek(f_main, 0, SEEK_END);
	size_t main_size = ftell(f_main);
	fseek(f_main, 0, SEEK_SET);

	uint8_t *main_content = malloc(main_size);
	if (!main_content) {
		fprintf(stderr, "failed to malloc\n");
		return 1;
	}

	fread(main_content, main_size, 1, f_main);
	printf("read file main of size %zu bytes\n", main_size);

	fclose(f_main);

	huffman_encoded encoded = huffman_encode(main_content, main_size);
	printf("encoded size: %zu, error: %d\n", encoded.encoded_size, encoded.error_code);

	huffman_decoded decoded = huffman_decode(encoded.encoded, encoded.encoded_size, encoded.code_length, encoded.original_size);
	printf("decoded size: %zu, error: %d\n", decoded.decoded_size, decoded.error_code);

	FILE *f_main2 = fopen("main2", "wab");
	if (!f_main2) {
		fprintf(stderr, "failed to create file main2: %s\n", strerror(errno));
		return 1;
	}

	fwrite(decoded.decoded, decoded.decoded_size, 1, f_main2);

	fclose(f_main2);

	return 0;
}