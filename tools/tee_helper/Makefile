CC ?= gcc
CFLAGS += -O2 -Wall -Wextra -I.
LDFLAGS += -lteec

all: tee_helper

tee_helper: tee_helper.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f tee_helper

