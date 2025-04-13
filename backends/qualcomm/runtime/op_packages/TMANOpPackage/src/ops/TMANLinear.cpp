//==============================================================================
// Auto Generated Code for TMANOpPackage
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"

#include "hvx_funcs.h"

BEGIN_PKG_OP_DEFINITION(PKG_TMANLinear);

static Qnn_Scalar_t sg_opDefaultGroup_SizeScalar = {.dataType = Qnn_DataType_t::QNN_DATATYPE_INT_32,
                                                   .int32Value = 64};
static Qnn_Param_t sg_opDefaultGroup_Size = {.paramType = QNN_PARAMTYPE_SCALAR,
                                            .scalarParam = sg_opDefaultGroup_SizeScalar};
static Qnn_Scalar_t sg_opDefaultBitsScalar = {.dataType = Qnn_DataType_t::QNN_DATATYPE_INT_32,
                                             .int32Value = 2};
static Qnn_Param_t sg_opDefaultBits = {.paramType = QNN_PARAMTYPE_SCALAR,
                                      .scalarParam = sg_opDefaultBitsScalar};
static Qnn_Scalar_t sg_opDefaultSymmetricScalar = {.dataType = Qnn_DataType_t::QNN_DATATYPE_INT_32,
                                                  .int32Value = 0};
static Qnn_Param_t sg_opDefaultSymmetric = {.paramType = QNN_PARAMTYPE_SCALAR,
                                           .scalarParam = sg_opDefaultSymmetricScalar};
static int called_times = 0;

template<typename TensorType>
GraphStatus tmanlinearImpl(TensorType& c,
                           const TensorType& l,
                           const TensorType& qweight,
                           const TensorType& scales,
                           const Int32Tensor& t_group_size,
                           const Int32Tensor& t_bits,
                           const Int32Tensor& t_symmetric);

static float tmanlinearCostFunc(const Op *op);

DEF_PACKAGE_OP((tmanlinearImpl<Tensor>), "TMANLinear")

DEF_TENSOR_PROPERTIES(Op("TMANLinear", "l", "qweight", "scales", "group_size", "bits", "symmetric"),
                      Flat("*", "l", "qweight", "scales"),
                      MainMemory("qweight", "scales", "group_size", "bits", "symmetric"),
                      Tcm("*", "l"))

DEF_PACKAGE_PARAM_ORDER("TMANLinear",
                        "group_size",
                        false,
                        &sg_opDefaultGroup_Size,
                        "bits",
                        false,
                        &sg_opDefaultBits,
                        "symmetric",
                        false,
                        &sg_opDefaultSymmetric)

template<typename TensorType>
GraphStatus tmanlinearImpl(TensorType& c,
                           const TensorType& l,
                           const TensorType& qweight,
                           const TensorType& scales,
                           const Int32Tensor& t_group_size,
                           const Int32Tensor& t_bits,
                           const Int32Tensor& t_symmetric)
{
  using LType = int16_t;
  using XType = __fp16;
  using CType = float;

  constexpr int32_t ACT_GROUP_SIZE = 256;
  constexpr int32_t LUT_G          = 4;
  constexpr int32_t LUT_SIZE       = 16;
  constexpr int32_t TILE_K         = 256;

  const int32_t group_size = ((const int32_t*)t_group_size.raw_data_const())[0];
  const int32_t bits       = ((const int32_t*)t_bits.raw_data_const())[0];
  // const bool zero_point    = ((const int32_t*)t_symmetric.raw_data_const())[0] == 0;
  const bool zero_point = true;

  const int32_t gemm_n = c.dims()[2];
  const int32_t gemm_m = qweight.dims()[2];
  const int32_t gemm_k = qweight.dims()[2] * qweight.dims()[3] * 32 / bits / gemm_m;

  const int32_t l_size  = gemm_k / LUT_G * LUT_SIZE;
  const int32_t ls_size = (ACT_GROUP_SIZE == -1) ? 1 : (gemm_k / ACT_GROUP_SIZE);

  const XType* x_buf  = (const XType*)l.raw_data_const();
  const LType* l_ptr  = (const LType*)(x_buf + gemm_k * gemm_n);
  const float* ls_ptr = (const float*)(l_ptr + l_size);
  const float* lb_ptr = ls_ptr + ls_size;

  const uint8_t* w_ptr = (const uint8_t*)qweight.raw_data_const();
  const XType* s_ptr   = (const XType*)scales.raw_data_const();
  CType* c_ptr         = (CType*)c.raw_data();

  if (zero_point && bits == 2 && group_size == 64)  // w2g64, symmetric=False
  {
    // hvx_tbl<LType, XType, CType, ACT_GROUP_SIZE, 64, true, 2, TILE_K, LUT_G>(gemm_m, gemm_k, gemm_n, l_ptr, ls_ptr, lb_ptr, w_ptr, s_ptr, c_ptr);
  }
  else if (!zero_point && bits == 4 && group_size == 128)  // w4g128, symmetric=True
  {
    // hvx_tbl<LType, XType, CType, ACT_GROUP_SIZE, 128, false, 4, TILE_K, LUT_G>(gemm_m, gemm_k, gemm_n, l_ptr, ls_ptr, lb_ptr, w_ptr, s_ptr, c_ptr);
  }
  else if (zero_point && bits == 4 && group_size == 128)  // w4g128, symmetric=False
  {
    hvx_tbl<LType, XType, CType, ACT_GROUP_SIZE, 128, true, 4, TILE_K, LUT_G>(gemm_m, gemm_k, gemm_n, l_ptr, ls_ptr, lb_ptr, w_ptr, s_ptr, c_ptr, called_times == 4370);
    // printf("linear: gemm_m: %d, gemm_k: %d, gemm_n: %d, group_size: %d, bits: %d, time: %d\n", gemm_m, gemm_k, gemm_n, group_size, bits, called_times);
  }
  else if (zero_point && bits == 4 && group_size == 64)  // w4g64, symmetric=False
  {
    // hvx_tbl<LType, XType, CType, ACT_GROUP_SIZE, 64, true, 4, TILE_K, LUT_G>(gemm_m, gemm_k, gemm_n, l_ptr, ls_ptr, lb_ptr, w_ptr, s_ptr, c_ptr);
  }
  else
  {
    return GraphStatus::ErrorDimensions;
  }
  // if (called_times == 4370)
  // {
  //   FILE* fp;
  //   fp = fopen("c.bin", "wb");
  //   fwrite(c_ptr, sizeof(CType), gemm_m * bits, fp);
  //   fclose(fp);
  //   printf("l_ptr: %p, ls_ptr: %p, lb_ptr: %p, w_ptr: %p, s_ptr: %p, c_ptr: %p\n", l_ptr, ls_ptr, lb_ptr, w_ptr, s_ptr, c_ptr);
  //   fp = fopen("w.bin", "wb");
  //   fwrite(w_ptr, sizeof(uint8_t), gemm_m * gemm_k * bits / 32, fp);
  //   fclose(fp);
  //   fp = fopen("s.bin", "wb");
  //   fwrite(s_ptr, sizeof(XType), gemm_m * gemm_k / group_size * (1 + zero_point), fp);
  //   fclose(fp);
  //   fp = fopen("l2.bin", "wb");
  //   fwrite(l_ptr, sizeof(LType), l_size, fp);
  //   fclose(fp);
  //   fp = fopen("ls2.bin", "wb");
  //   fwrite(ls_ptr, sizeof(float), ls_size, fp);
  //   fclose(fp);
  //   fp = fopen("lb2.bin", "wb");
  //   fwrite(lb_ptr, sizeof(float), gemm_k / group_size, fp);
  //   fclose(fp);
  // }
  called_times += 1;

  return GraphStatus::Success;
}

__attribute__((unused)) static float tmanlinearCostFunc(const Op *op)
{
  float cost = 0.0;  // add cost computation here
  return cost;
}

END_PKG_OP_DEFINITION(PKG_TMANLinear);
