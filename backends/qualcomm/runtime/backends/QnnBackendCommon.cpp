/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/backends/QnnBackendCommon.h>
#include <sstream>
namespace executorch {
namespace backends {
namespace qnn {

using executorch::runtime::Error;

namespace {
void split(
    std::vector<std::string>& splitString,
    const std::string& tokenizedString,
    const char separator) {
  splitString.clear();
  std::istringstream tokenizedStringStream(tokenizedString);
  while (!tokenizedStringStream.eof()) {
    std::string value;
    getline(tokenizedStringStream, value, separator);
    if (!value.empty()) {
      splitString.push_back(value);
    }
  }
}
} // namespace

QnnBackend::~QnnBackend() {
  const QnnInterface& qnn_interface = implementation_.GetQnnInterface();
  Qnn_ErrorHandle_t error = QNN_SUCCESS;
  if (nullptr != handle_) {
    QNN_EXECUTORCH_LOG_INFO("Destroy Qnn backend");
    error = qnn_interface.qnn_backend_free(handle_);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Failed to free QNN "
          "backend_handle. Backend "
          "ID %u, error %d",
          qnn_interface.GetBackendId(),
          QNN_GET_ERROR_CODE(error));
    }
    handle_ = nullptr;
  }
}

Error QnnBackend::Configure() {
  // create qnn backend
  const QnnInterface& qnn_interface = implementation_.GetQnnInterface();
  Qnn_ErrorHandle_t error = QNN_SUCCESS;

  std::vector<const QnnBackend_Config_t*> temp_backend_config;
  ET_CHECK_OR_RETURN_ERROR(
      MakeConfig(temp_backend_config) == Error::Ok,
      Internal,
      "Fail to make backend config.");

  error = qnn_interface.qnn_backend_create(
      logger_->GetHandle(),
      temp_backend_config.empty() ? nullptr : temp_backend_config.data(),
      &handle_);
  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Failed to create "
        "backend_handle for Backend "
        "ID %u, error=%d",
        qnn_interface.GetBackendId(),
        QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }

  // TODO: Expose API to options in QnnManager later
  std::string opPackagePaths = "/data/local/tmp/llama/libQnnTMANOpPackage.so:TMANOpPackageInterfaceProvider:HTP";
  if (const char* env_p = std::getenv("QNN_OP_PACKAGE_PATHS")) {
    opPackagePaths = env_p;
  }
  std::vector<std::string> m_opPackagePaths;
  split(m_opPackagePaths, opPackagePaths, ',');

  const size_t pathIdx = 0;
  const size_t interfaceProviderIdx = 1;
  for (auto const& opPackagePath : m_opPackagePaths) {
    std::vector<std::string> opPackage;
    split(opPackage, opPackagePath, ':');
    const char* target = nullptr;
    const size_t targetIdx = 2;
    if (opPackage.size() != 2 && opPackage.size() != 3) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Malformed opPackageString provided: %s", opPackagePath.c_str());
      return Error::Internal;
    }
    if (opPackage.size() == 3) {
      target = opPackage[targetIdx].c_str();
    }
    error = qnn_interface.qnn_backend_register_op_package(
        handle_, opPackage[pathIdx].c_str(), opPackage[interfaceProviderIdx].c_str(), target);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Failed to register "
          "op package %s for Backend "
          "ID %u, error=%d",
          opPackage[pathIdx].c_str(),
          qnn_interface.GetBackendId(),
          QNN_GET_ERROR_CODE(error));
      return Error::Internal;
    }
    QNN_EXECUTORCH_LOG_INFO(
        "Registered Op Package: %s and interface provider: %s",
        opPackage[pathIdx].c_str(),
        opPackage[interfaceProviderIdx].c_str());
  }
  return Error::Ok;
}

Error QnnBackend::VerifyQNNSDKVersion() {
  const QnnInterface& qnn_interface = implementation_.GetQnnInterface();

  Qnn_ApiVersion_t qnn_version = {QNN_VERSION_INIT};
  Qnn_ErrorHandle_t error =
      qnn_interface.qnn_backend_get_api_version(&qnn_version);
  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR("Failed to get Qnn API version.");
    return Error::Internal;
  }

  Qnn_ApiVersion_t expected_version = {QNN_VERSION_INIT};
  expected_version.coreApiVersion.major = QNN_API_VERSION_MAJOR;
  expected_version.coreApiVersion.minor = QNN_API_VERSION_MINOR;
  expected_version.coreApiVersion.patch = QNN_API_VERSION_PATCH;
  expected_version.backendApiVersion = QNN_VERSION_INIT;
  if (qnn_interface.GetBackendId() == QNN_BACKEND_ID_SAVER) {
    expected_version.backendApiVersion.major = QNN_SAVER_API_VERSION_MAJOR;
    expected_version.backendApiVersion.minor = QNN_SAVER_API_VERSION_MINOR;
    expected_version.backendApiVersion.patch = QNN_SAVER_API_VERSION_PATCH;
  } else {
    expected_version.backendApiVersion = GetExpectedBackendVersion();
  }
  const char* backend_type = EnumNameQnnExecuTorchBackendType(
      static_cast<QnnExecuTorchBackendType>(qnn_interface.GetBackendId()));

  Error status = VersionChecker(
      qnn_version.coreApiVersion, expected_version.coreApiVersion, "Qnn API");
  if (status == Error::Ok) {
    status = VersionChecker(
        qnn_version.backendApiVersion,
        expected_version.backendApiVersion,
        backend_type);
  }

  return status;
}

Error QnnBackend::VersionChecker(
    const Qnn_Version_t& qnn_version,
    const Qnn_Version_t& expected,
    const std::string& prefix) {
  if (qnn_version.major != expected.major) {
    QNN_EXECUTORCH_LOG_ERROR(
        "%s version %u.%u.%u is not supported. "
        "The minimum supported version is %u.%u.%u. Please make "
        "sure you have the correct backend library version.",
        prefix.c_str(),
        qnn_version.major,
        qnn_version.minor,
        qnn_version.patch,
        expected.major,
        expected.minor,
        expected.patch);
    return Error::Internal;
  }
  if (qnn_version.major == QNN_API_VERSION_MAJOR &&
      qnn_version.minor < expected.minor) {
    QNN_EXECUTORCH_LOG_WARN(
        "%s version %u.%u.%u is mismatched. "
        "The minimum supported version is %u.%u.%u. Please make "
        "sure you have the correct backend library version.",
        prefix.c_str(),
        qnn_version.major,
        qnn_version.minor,
        qnn_version.patch,
        expected.major,
        expected.minor,
        expected.patch);
  }
  if ((qnn_version.major == QNN_API_VERSION_MAJOR &&
       qnn_version.minor > expected.minor)) {
    QNN_EXECUTORCH_LOG_WARN(
        "%s version %u.%u.%u is used. "
        "The version is tested against %u.%u.%u.",
        prefix.c_str(),
        qnn_version.major,
        qnn_version.minor,
        qnn_version.patch,
        expected.major,
        expected.minor,
        expected.patch);
  }
  return Error::Ok;
}
} // namespace qnn
} // namespace backends
} // namespace executorch
