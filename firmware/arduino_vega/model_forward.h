#ifndef MODEL_FORWARD_H
#define MODEL_FORWARD_H

#include <stdint.h>
#include <stddef.h>

#define INPUT_LEN     256
#define NUM_CLASSES   5

/* Runs the quantized MultiScale_SE_CNN_Tiny forward pass on one
 * Q8.8 fixed-point ECG window of length 256.
 *
 * in:      Q8.8 int16 samples, length INPUT_LEN
 * logits:  raw (pre-softmax) Q8.8 class scores, length NUM_CLASSES —
 *          softmax is monotonic, so argmax(logits) == argmax(softmax(logits));
 *          pass NULL if you only need the class index.
 *
 * Returns the predicted class index (0..4), matching CLASS_NAMES =
 * ['Normal(N)','SVEB(S)','VEB(V)','Fusion(F)','Unknown(Q)'].
 */
int model_forward(const int16_t in[INPUT_LEN], int16_t logits[NUM_CLASSES]);

/* Total bytes of static activation buffers the kernel holds (excludes the
 * const weight tables, which live in flash/rodata). Useful when budgeting
 * SRAM on the VEGA target. */
size_t model_forward_ram_bytes(void);

#endif /* MODEL_FORWARD_H */
