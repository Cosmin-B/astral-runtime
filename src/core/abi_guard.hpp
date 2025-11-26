#pragma once

#include "error.hpp"

// Translate unexpected C++ exceptions before control returns across the C ABI.
//
// Notes:
// - This does not remove exception support from the binary; it catches and translates.
// - When compiled with exceptions disabled (-fno-exceptions), these macros become no-ops.

#if (defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(_CPPUNWIND)) && ASTRAL_NO_THROW_ABI
  #define ASTRAL_ABI_TRY_BEGIN try { ::astral::core::ErrorScope _astral_err_scope;
  #define ASTRAL_ABI_CATCH_END_ERR(ret_err)                                    \
    } catch (...) {                                                           \
      astral::core::set_last_errorf("Unhandled C++ exception crossed ABI");   \
      return (ret_err);                                                       \
    }
  #define ASTRAL_ABI_CATCH_END_I32(ret_i32)                                    \
    } catch (...) {                                                           \
      astral::core::set_last_errorf("Unhandled C++ exception crossed ABI");   \
      return (ret_i32);                                                       \
    }
  #define ASTRAL_ABI_CATCH_END_VOID()                                          \
    } catch (...) {                                                           \
      astral::core::set_last_errorf("Unhandled C++ exception crossed ABI");   \
      return;                                                                 \
    }
  #define ASTRAL_ABI_CATCH_END_CSTR()                                          \
    } catch (...) {                                                           \
      astral::core::set_last_errorf("Unhandled C++ exception crossed ABI");   \
      return "Unknown error";                                                 \
    }
#else
  #define ASTRAL_ABI_TRY_BEGIN { ::astral::core::ErrorScope _astral_err_scope;
  #define ASTRAL_ABI_CATCH_END_ERR(ret_err) }
  #define ASTRAL_ABI_CATCH_END_I32(ret_i32) }
  #define ASTRAL_ABI_CATCH_END_VOID() }
  #define ASTRAL_ABI_CATCH_END_CSTR() }
#endif
