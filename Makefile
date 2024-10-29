default: test

suitesparseqr.so: suitesparseqr.c
	gcc suitesparseqr.c -shared -o suitesparseqr.so `pkg-config --cflags --libs python3 numpy` -fPIC

test: suitesparseqr.so
	python test.py

clean:
	rm *.so