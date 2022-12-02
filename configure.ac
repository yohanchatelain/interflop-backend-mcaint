AC_INIT([interflop-backend-mcaint],[1.0],[interflop.project@gmail.com])
AM_SILENT_RULES([yes])
AC_CONFIG_AUX_DIR(autoconf)
AM_INIT_AUTOMAKE([subdir-objects -Wall -Werror foreign])
AC_CONFIG_MACRO_DIRS([m4])
AC_PROG_CC
AM_PROG_AR
AC_CONFIG_HEADERS([config.h])
LT_INIT
AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_MAKE_SET

AC_ARG_ENABLE(wall, AS_HELP_STRING([--enable-wall],[Enable -Wall compilation flag]), [WALL_CFLAGS="yes"])
AM_CONDITIONAL([WALL_CFLAGS], [test "x$WALL_CFLAGS" = "xyes"])
if test "x$WALL_CFLAGS" = "xyes"; then
   AC_DEFINE([WALL_CFLAGS], [],  ["Enable -Wall compilation flag"])
   AC_MSG_NOTICE([-Wall is set])
fi

AC_ARG_ENABLE(rng-thread-safe, AS_HELP_STRING([--enable-rng-thread-safe],[Enable Thread-Safe RNG]), [RNG_THREAD_SAFE="$withval"], [RNG_THREAD_SAFE="yes"])
AM_CONDITIONAL([RNG_THREAD_SAFE], [test "x$RNG_THREAD_SAFE" = "xyes"])
if test "x$RNG_THREAD_SAFE" = "xyes"; then
   AC_DEFINE([RNG_THREAD_SAFE], [],  ["Enable --rng-thread-safe"])
else
   AC_MSG_NOTICE([--rng-thread-safe is disabled])
fi

AX_INTERFLOP_STDLIB()

AC_CONFIG_FILES([
 Makefile
])
AC_OUTPUT