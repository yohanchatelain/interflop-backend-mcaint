SUBDIRS=common
lib_LTLIBRARIES = libinterflop_mca_int.la
libinterflop_mca_int_la_SOURCES = interflop_mca_int.c common/logger.c common/options.c
libinterflop_mca_int_la_CFLAGS = -DBACKEND_HEADER="interflop_mca_int" -O3 -I@INTERFLOP_STDLIB_PATH@/include/
if WALL_CFLAGS
libinterflop_mca_int_la_CFLAGS += -Wall -Wextra
endif
libinterflop_mca_int_la_LDFLAGS = -lm
libinterflop_mca_int_la_LIBADD = @INTERFLOP_STDLIB_PATH@/lib/libinterflop_rng.la common/libinterflop_fma.la
library_includedir =$(includedir)/
