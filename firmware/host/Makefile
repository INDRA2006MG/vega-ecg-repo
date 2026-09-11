CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra -Wpedantic -std=c11

CORE    = model_forward.c weights.c

.PHONY: all verify real rpeak fusion clean

all: run_vectors test_sanity

run_vectors: run_vectors.c test_vectors_synth.h $(CORE) model_forward.h weights.h
	$(CC) $(CFLAGS) -o $@ run_vectors.c $(CORE)

test_sanity: test_sanity.c $(CORE) model_forward.h weights.h
	$(CC) $(CFLAGS) -o $@ test_sanity.c $(CORE) -lm

test_vectors_synth.h test_inputs_q88.npy: make_vectors.py
	python3 make_vectors.py

# Full verification: C kernel vs NumPy fixed-point sim (must be exact)
# and fixed-point vs float (quantization drift).
verify: run_vectors test_inputs_q88.npy
	./run_vectors > c_out.txt
	python3 compare.py

# Regenerate weights.c/weights.h from the Colab-exported weights_keras.h
weights.c weights.h: weights_keras.h convert_weights.py
	python3 convert_weights.py

# Validate against the real MIT-BIH beats exported from Colab
real: run_real
	./run_real
	./run_real --dump > c_real.txt
	python3 compare_real.py

run_real: run_real.c test_vectors.h $(CORE) model_forward.h weights.h
	$(CC) $(CFLAGS) -o $@ run_real.c $(CORE)

# R-peak detector: synthetic smoke test (real scoring needs rpeak_vectors.h)
rpeak: test_rpeak
	./test_rpeak

test_rpeak: test_rpeak.c rpeak.c rpeak.h
	$(CC) $(CFLAGS) -o $@ test_rpeak.c rpeak.c -lm

# Sensor-fusion severity rules: precedence checks (see sensor_thresholds.h)
fusion: test_fusion
	./test_fusion

test_fusion: test_fusion.c sensor_thresholds.h
	$(CC) $(CFLAGS) -o $@ test_fusion.c

clean:
	rm -f run_vectors run_real test_sanity test_rpeak test_fusion c_out.txt c_real.txt
