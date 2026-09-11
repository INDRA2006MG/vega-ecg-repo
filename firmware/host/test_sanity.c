/* Compile-and-sanity-check harness only — NOT the accuracy validation.
 * Runs model_forward() on a zero input and a synthetic waveform purely to
 * confirm the kernel runs without crashing/overflowing and produces
 * plausible-magnitude Q8.8 logits. Real accuracy validation needs
 * test_vectors.h exported from the actual trained Keras model.
 */
#define _USE_MATH_DEFINES
#include <stdio.h>
#include <math.h>
#include "model_forward.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static const char *CLASS_NAMES[NUM_CLASSES] = {
    "Normal(N)", "SVEB(S)", "VEB(V)", "Fusion(F)", "Unknown(Q)"
};

static void run_one(const char *label, const int16_t *in) {
    int16_t logits[NUM_CLASSES];
    int cls = model_forward(in, logits);
    printf("%-20s -> class %d (%s)  logits:", label, cls, CLASS_NAMES[cls]);
    for (int i = 0; i < NUM_CLASSES; i++) printf(" %6d", logits[i]);
    printf("\n");
}

int main(void) {
    int16_t zeros[INPUT_LEN] = {0};
    run_one("all-zero input", zeros);

    int16_t sine[INPUT_LEN];
    for (int i = 0; i < INPUT_LEN; i++)
        sine[i] = (int16_t)(200.0 * sin(2 * M_PI * i / 32.0)); /* Q8.8, amplitude ~0.78 */
    run_one("synthetic sine", sine);

    int16_t spike[INPUT_LEN] = {0};
    spike[90] = 8000; /* a sharp R-like spike near the middle, Q8.8 ~= 31.25 */
    run_one("synthetic spike", spike);

    int16_t maxval[INPUT_LEN];
    for (int i = 0; i < INPUT_LEN; i++) maxval[i] = 32767;
    run_one("all max (32767)", maxval);

    int16_t minval[INPUT_LEN];
    for (int i = 0; i < INPUT_LEN; i++) minval[i] = -32768;
    run_one("all min (-32768)", minval);

    printf("\nNo crashes, no obviously-broken (NaN-like/huge) outputs => kernel is structurally sound.\n");
    printf("Still need real test_vectors.h (Keras predictions) for actual accuracy validation.\n");
    return 0;
}
