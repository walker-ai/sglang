CROSS_COMPILE ?=
CC := $(CROSS_COMPILE)gcc

TEEC_EXPORT ?=
PKG_CONFIG ?= pkg-config

ifeq ($(strip $(TEEC_EXPORT)),)
CFLAGS += -O2 -Wall -Wextra $(shell $(PKG_CONFIG) --cflags libteec)
LDFLAGS += $(shell $(PKG_CONFIG) --libs libteec)
else
CFLAGS += -O2 -Wall -Wextra -I$(TEEC_EXPORT)/include
LDFLAGS += -L$(TEEC_EXPORT)/lib -lteec
endif

all: tee_helper

tee_helper: tee_helper.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f tee_helper
