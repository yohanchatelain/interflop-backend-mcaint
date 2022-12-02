/*****************************************************************************\
 *                                                                           *\
 *  This file is part of the Verificarlo project,                            *\
 *  under the Apache License v2.0 with LLVM Exceptions.                      *\
 *  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.                 *\
 *  See https://llvm.org/LICENSE.txt for license information.                *\
 *                                                                           *\
 *  Copyright (c) 2019-2022                                                  *\
 *     Verificarlo Contributors                                              *\
 *                                                                           *\
 ****************************************************************************/

#ifndef __INTERFLOP_MCAINT_H__
#define __INTERFLOP_MCAINT_H__

#include "interflop-stdlib/interflop_stdlib.h"

#define INTERFLOP_MCAINT_API(name) interflop_mcaint_##name

/* define default environment variables and default parameters */
#define MCA_PRECISION_BINARY32_MIN 1
#define MCA_PRECISION_BINARY64_MIN 1
#define MCA_PRECISION_BINARY32_MAX DOUBLE_PMAN_SIZE
#define MCA_PRECISION_BINARY64_MAX QUAD_PMAN_SIZE
#define MCA_PRECISION_BINARY32_DEFAULT FLOAT_PREC
#define MCA_PRECISION_BINARY64_DEFAULT DOUBLE_PREC
#define MCA_MODE_DEFAULT mcamode_mca

/* define the available MCA modes of operation */
typedef enum {
  mcamode_ieee,
  mcamode_mca,
  mcamode_pb,
  mcamode_rr,
  _mcamode_end_
} mcamode;

/* Interflop context */
typedef struct {
  IBool relErr;
  IBool absErr;
  IBool daz;
  IBool ftz;
  IBool choose_seed;
  mcamode mode;
  int binary32_precision;
  int binary64_precision;
  int absErr_exp;
  float sparsity;
  IUint64_t seed;
} mcaint_context_t;

#endif /* __INTERFLOP_MCAINT_H__ */