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

BEGIN_PKG_OP_DEFINITION(PKG_TMANPrecompute);

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
GraphStatus tmanprecomputeImpl(TensorType& l,
                               const TensorType& x,
                               const Int32Tensor& t_group_size,
                               const Int32Tensor& t_bits,
                               const Int32Tensor& t_symmetric);

static float tmanprecomputeCostFunc(const Op *op);

DEF_PACKAGE_OP((tmanprecomputeImpl<Tensor>), "TMANPrecompute")

DEF_TENSOR_PROPERTIES(Op("TMANPrecompute", "x", "group_size", "bits", "symmetric"),
                      Flat("*", "x"),
                      MainMemory("group_size", "bits", "symmetric"),
                      Tcm("*", "x"))

DEF_PACKAGE_PARAM_ORDER("TMANPrecompute",
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
GraphStatus tmanprecomputeImpl(TensorType& l,
                               const TensorType& x,
                               const Int32Tensor& t_group_size,
                               const Int32Tensor& t_bits,
                               const Int32Tensor& t_symmetric)
{
  using LType = int16_t;
  using XType = __fp16;

  constexpr int32_t ACT_GROUP_SIZE = 256;
  constexpr int32_t LUT_G          = 4;
  constexpr int32_t LUT_SIZE       = 16;

  const int32_t gemm_k = x.dims()[3];
  const int32_t gemm_n = x.dims()[2];

  const int32_t group_size = ((const int32_t*)t_group_size.raw_data_const())[0];
  // const bool zero_point    = ((const int32_t*)t_symmetric.raw_data_const())[0] == 0;
  const bool zero_point = true;

  const int32_t l_size  = gemm_k / LUT_G * LUT_SIZE;
  const int32_t ls_size = (ACT_GROUP_SIZE == -1) ? 1 : (gemm_k / ACT_GROUP_SIZE);

  XType* x_buf  = (XType*)l.raw_data();
  LType* l_ptr  = (LType*)(x_buf + gemm_k * gemm_n);
  float* ls_ptr = (float*)(l_ptr + l_size);
  float* lb_ptr = ls_ptr + ls_size;

  const XType* x_ptr = (const XType*)x.raw_data_const();

  if (x.get_dtype() == DType::QUInt16)
  {
    const float x_scale = x.get_interface_scale();
    const int x_offset   = x.get_interface_offset();

    HVX_Vector x_offset_vec = Q6_V_vsplat_R(x_offset);
    HVX_Vector x_scale_vec  = Q6_V_vsplat_R(_fp32_to_bits(x_scale));
    for (int32_t i = 0; i < gemm_k * gemm_n; i += VLEN / sizeof(uint16_t))
    {
      HVX_Vector x_vec = vmem(x_ptr + i);
      HVX_VectorPair x_vec_pair = Q6_Wuw_vzxt_Vuh(x_vec);
      HVX_Vector x_vec_lo = Q6_V_lo_W(x_vec_pair);
      HVX_Vector x_vec_hi = Q6_V_hi_W(x_vec_pair);
      x_vec_lo = Q6_Vw_vsub_VwVw(x_vec_lo, x_offset_vec);
      x_vec_hi = Q6_Vw_vsub_VwVw(x_vec_hi, x_offset_vec);
      x_vec_lo = Q6_Vsf_equals_Vw(x_vec_lo);
      x_vec_hi = Q6_Vsf_equals_Vw(x_vec_hi);
      x_vec_lo = Q6_Vqf32_vmpy_VsfVsf(x_vec_lo, x_scale_vec);
      x_vec_hi = Q6_Vqf32_vmpy_VsfVsf(x_vec_hi, x_scale_vec);
      x_vec = Q6_Vhf_equals_Vqf16(x_vec);
      x_vec = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(x_vec_hi, x_vec_lo));
      vmem(x_buf + i) = x_vec;
    }
  }

  if (zero_point && group_size == 64)  // w2g64, symmetric=False
  {
    hvx_lut_ctor<LType, XType, ACT_GROUP_SIZE, 64, true, LUT_G>(gemm_k, gemm_n, x_buf, l_ptr, ls_ptr, lb_ptr);
  }
  else if (!zero_point && group_size == 128)  // w4g128, symmetric=True
  {
    hvx_lut_ctor<LType, XType, ACT_GROUP_SIZE, 128, false, LUT_G>(gemm_k, gemm_n, x_buf, l_ptr, ls_ptr, lb_ptr);
  }
  else if (zero_point && group_size == 128)  // w4g128, symmetric=False
  {
    hvx_lut_ctor<LType, XType, ACT_GROUP_SIZE, 128, true, LUT_G>(gemm_k, gemm_n, x_buf, l_ptr, ls_ptr, lb_ptr);
  }
  else
  {
    return GraphStatus::ErrorDimensions;
  }

  return GraphStatus::Success;
}

__attribute__((unused)) static float tmanprecomputeCostFunc(const Op *op)
{
  float cost = 0.0;  // add cost computation here
  return cost;
}

END_PKG_OP_DEFINITION(PKG_TMANPrecompute);
