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

template<typename TensorType>
GraphStatus tmanlinearImpl(TensorType& y,
                           TensorType& scratch_buffer,
                           const TensorType& x,
                           const TensorType& qweight,
                           const TensorType& scales,
                           const Int32Tensor& t_group_size,
                           const Int32Tensor& t_bits,
                           const Int32Tensor& t_symmetric);

static float tmanlinearCostFunc(const Op *op);

DEF_PACKAGE_OP((tmanlinearImpl<Tensor>), "TMANLinear")

// Tcm("y") results in [ERROR] [Qnn ExecuTorch]: graph_prepare.cc:217:ERROR:could not create op: q::Add.tcm
// Reason: embedding (Gather) outputs are in MainMemory
//         but TMANLinear outputs are in Tcm
//         add(embedding, TMANLinear) thus causes a conflict
// TODO:
// - implement custom TMANOpPackage::Add
DEF_TENSOR_PROPERTIES(Op("TMANLinear", "x", "qweight", "scales", "group_size", "bits", "symmetric"),
                      Outputs("y", "scratch"),
                      Flat("y", "scratch", "x", "qweight", "scales"),
                      MainMemory("y", "qweight", "scales", "group_size", "bits", "symmetric"),
                      Tcm("scratch", "x"))

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
GraphStatus tmanlinearImpl(TensorType& y,
                           TensorType& scratch_buffer,
                           const TensorType& x,
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

  const int32_t gemm_m = y.dims()[2];
  const int32_t gemm_k = x.dims()[2];
  const int32_t gemm_n = x.dims()[3];

  const int32_t group_size = ((const int32_t*)t_group_size.raw_data_const())[0];
  const int32_t bits       = ((const int32_t*)t_bits.raw_data_const())[0];
  const bool zero_point    = ((const int32_t*)t_symmetric.raw_data_const())[0] == 0;

  const int32_t l_size  = gemm_k / LUT_G * LUT_SIZE;
  const int32_t ls_size = (ACT_GROUP_SIZE == -1) ? 1 : (gemm_k / ACT_GROUP_SIZE);
  const int32_t lb_size = (group_size == 0) ? 1 : (gemm_k / group_size);

  LType* l_ptr  = (LType*)scratch_buffer.raw_data();
  float* ls_ptr = (float*)(l_ptr + l_size);
  float* lb_ptr = ls_ptr + ls_size;
  CType* c_ptr  = (CType*)(lb_ptr + lb_size);

  const XType* x_ptr   = (const XType*)x.raw_data_const();
  const uint8_t* w_ptr = (const uint8_t*)qweight.raw_data_const();
  const XType* s_ptr   = (const XType*)scales.raw_data_const();
  XType* y_ptr         = (XType*)y.raw_data();

  // TODO: fuse with lut_ctor
  // if (x.get_dtype() == DType::QUInt16)
  // {
  //   const __fp16 x_scale = x.get_interface_scale();
  //   const int x_offset   = x.get_interface_offset();

  //   HVX_Vector x_offset_vec = Q6_Vh_vsplat_R((int16_t)x_offset);
  //   HVX_Vector x_scale_vec  = Q6_Vh_vsplat_R(_fp16_to_bits(&x_scale));
  //   for (int32_t i = 0; i < gemm_k * gemm_n; i += VLEN / sizeof(XType))
  //   {
  //     HVX_Vector x_vec = vmem(x_ptr + i);
  //     x_vec = Q6_Vh_vsub_VhVh_sat(x_vec, x_offset_vec);
  //     x_vec = Q6_Vqf16_vmpy_VhfVhf(x_vec, x_scale_vec);
  //     x_vec = Q6_Vhf_equals_Vqf16(x_vec);
  //     vmem(x_ptr + i) = x_vec;
  //   }
  // }

  if (true)
  // if (zero_point && bits == 2 && group_size == 64)  // w2g64, symmetric=False
  {
    hvx_lut_ctor<LType, XType, ACT_GROUP_SIZE, 64, true, 2, LUT_G>(gemm_k, gemm_n, x_ptr, l_ptr, ls_ptr, lb_ptr);
    hvx_tbl<LType, XType, CType, ACT_GROUP_SIZE, 64, true, 2, TILE_K, LUT_G>(gemm_m, gemm_k, gemm_n, l_ptr, ls_ptr, lb_ptr, w_ptr, s_ptr, y_ptr, c_ptr);
  }
  else if (!zero_point && bits == 4 && group_size == 128)  // w4g128, symmetric=True
  {
    hvx_lut_ctor<LType, XType, ACT_GROUP_SIZE, 128, false, 4, LUT_G>(gemm_k, gemm_n, x_ptr, l_ptr, ls_ptr, lb_ptr);
    hvx_tbl<LType, XType, CType, ACT_GROUP_SIZE, 128, false, 4, TILE_K, LUT_G>(gemm_m, gemm_k, gemm_n, l_ptr, ls_ptr, lb_ptr, w_ptr, s_ptr, y_ptr, c_ptr);
  }
  else if (zero_point && bits == 4 && group_size == 128)  // w4g128, symmetric=False
  {
    hvx_lut_ctor<LType, XType, ACT_GROUP_SIZE, 128, true, 4, LUT_G>(gemm_k, gemm_n, x_ptr, l_ptr, ls_ptr, lb_ptr);
    hvx_tbl<LType, XType, CType, ACT_GROUP_SIZE, 128, true, 4, TILE_K, LUT_G>(gemm_m, gemm_k, gemm_n, l_ptr, ls_ptr, lb_ptr, w_ptr, s_ptr, y_ptr, c_ptr);
  }
  else if (zero_point && bits == 4 && group_size == 64)  // w4g64, symmetric=False
  {
    hvx_lut_ctor<LType, XType, ACT_GROUP_SIZE, 64, true, 4, LUT_G>(gemm_k, gemm_n, x_ptr, l_ptr, ls_ptr, lb_ptr);
    hvx_tbl<LType, XType, CType, ACT_GROUP_SIZE, 64, true, 4, TILE_K, LUT_G>(gemm_m, gemm_k, gemm_n, l_ptr, ls_ptr, lb_ptr, w_ptr, s_ptr, y_ptr, c_ptr);
  }
  else
  {
    return GraphStatus::ErrorDimensions;
  }

  // TODO: fuse with tbl
  // if (y.get_dtype() == DType::QUInt16)
  // {
  //   const __fp16 y_scale_recip = y.interface().get_scale_recip();
  //   const int y_offset         = y.get_interface_offset();

  //   HVX_Vector y_offset_vec      = Q6_Vh_vsplat_R((int16_t)y_offset);
  //   HVX_Vector y_scale_recip_vec = Q6_Vh_vsplat_R(_fp16_to_bits(&y_scale_recip));
  //   for (int32_t i = 0; i < gemm_m * gemm_n; i += VLEN / sizeof(XType))
  //   {
  //     HVX_Vector y_vec = vmem(y_ptr + i);
  //     y_vec = Q6_Vqf16_vmpy_VhfVhf(y_vec, y_scale_recip_vec);
  //     y_vec = Q6_Vh_equals_Vhf(Q6_Vhf_equals_Vqf16(y_vec));
  //     y_vec = Q6_Vh_vadd_VhVh_sat(y_vec, y_offset_vec);
  //     vmem(y_ptr + i) = y_vec;
  //   }
  // }

  return GraphStatus::Success;
}

__attribute__((unused)) static float tmanlinearCostFunc(const Op *op)
{
  float cost = 0.0;  // add cost computation here
  return cost;
}

END_PKG_OP_DEFINITION(PKG_TMANLinear);
