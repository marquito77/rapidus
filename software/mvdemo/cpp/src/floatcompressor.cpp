#include "floatcompressor.h"

void floats2halfs(const float* i_floats, uint16_t*  o_halfs,  int i_numFloats)
{
	Float16Compressor fc;

	for (int i=0; i<i_numFloats; ++i) {
		o_halfs[i] = fc.compress(i_floats[i]);
	}
}

void halfs2floats(const uint16_t*  i_halfs,  float* o_floats, int i_numFloats)
{
	Float16Compressor fc;

	for (int i=0; i<i_numFloats; ++i) {
		o_floats[i] = fc.decompress(i_halfs[i]);
	}
}
