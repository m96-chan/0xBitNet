{-# LANGUAGE CApiFFI #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE PatternSynonyms #-}

-- | Low-level FFI bindings to @liboxbitnet_ffi@.
module Foreign.OxBitNet.Raw
  ( -- * Opaque handle
    OxBitNet

    -- * Enums
  , LoadPhase
  , pattern PhaseDownload
  , pattern PhaseParse
  , pattern PhaseUpload
  , LogLevel
  , pattern LogTrace
  , pattern LogDebug
  , pattern LogInfo
  , pattern LogWarn
  , pattern LogError

    -- * Structs
  , CGenerateOptions(..)
  , CLoadProgress(..)
  , CLoadOptions(..)
  , CChatMessage(..)

    -- * Callback types
  , TokenFn
  , ProgressFn
  , LogFn
  , mkTokenFn
  , mkProgressFn
  , mkLogFn

    -- * FFI functions
  , oxbitnet_set_logger
  , oxbitnet_default_generate_options
  , oxbitnet_default_load_options
  , oxbitnet_load
  , oxbitnet_free
  , oxbitnet_generate
  , oxbitnet_chat
  , oxbitnet_error_message
  ) where

import Foreign
import Foreign.C

#include "oxbitnet.h"

-- --------------------------------------------------------------------
-- Opaque handle
-- --------------------------------------------------------------------

-- | Opaque handle to a loaded BitNet model (never constructed in Haskell).
data OxBitNet

-- --------------------------------------------------------------------
-- Enums
-- --------------------------------------------------------------------

-- | Load progress phase.
newtype LoadPhase = LoadPhase CInt
  deriving (Eq, Ord, Show, Storable)

pattern PhaseDownload, PhaseParse, PhaseUpload :: LoadPhase
pattern PhaseDownload = LoadPhase #{const Download}
pattern PhaseParse    = LoadPhase #{const Parse}
pattern PhaseUpload   = LoadPhase #{const Upload}

-- | Log level.
newtype LogLevel = LogLevel Word8
  deriving (Eq, Ord, Show, Storable)

pattern LogTrace, LogDebug, LogInfo, LogWarn, LogError :: LogLevel
pattern LogTrace = LogLevel #{const Trace}
pattern LogDebug = LogLevel #{const Debug}
pattern LogInfo  = LogLevel #{const Info}
pattern LogWarn  = LogLevel #{const Warn}
pattern LogError = LogLevel #{const Error}

-- --------------------------------------------------------------------
-- Structs
-- --------------------------------------------------------------------

-- | @OxBitNetGenerateOptions@
data CGenerateOptions = CGenerateOptions
  { cgoMaxTokens    :: !#{type uintptr_t}
  , cgoTemperature  :: !CFloat
  , cgoTopK         :: !#{type uintptr_t}
  , cgoRepeatPenalty :: !CFloat
  , cgoRepeatLastN  :: !#{type uintptr_t}
  } deriving (Show)

instance Storable CGenerateOptions where
  sizeOf    _ = #{size    OxBitNetGenerateOptions}
  alignment _ = #{alignment OxBitNetGenerateOptions}
  peek p = CGenerateOptions
    <$> #{peek OxBitNetGenerateOptions, max_tokens}     p
    <*> #{peek OxBitNetGenerateOptions, temperature}    p
    <*> #{peek OxBitNetGenerateOptions, top_k}          p
    <*> #{peek OxBitNetGenerateOptions, repeat_penalty} p
    <*> #{peek OxBitNetGenerateOptions, repeat_last_n}  p
  poke p v = do
    #{poke OxBitNetGenerateOptions, max_tokens}     p (cgoMaxTokens v)
    #{poke OxBitNetGenerateOptions, temperature}    p (cgoTemperature v)
    #{poke OxBitNetGenerateOptions, top_k}          p (cgoTopK v)
    #{poke OxBitNetGenerateOptions, repeat_penalty} p (cgoRepeatPenalty v)
    #{poke OxBitNetGenerateOptions, repeat_last_n}  p (cgoRepeatLastN v)

-- | @OxBitNetLoadProgress@
data CLoadProgress = CLoadProgress
  { clpPhase    :: !LoadPhase
  , clpLoaded   :: !Word64
  , clpTotal    :: !Word64
  , clpFraction :: !CDouble
  } deriving (Show)

instance Storable CLoadProgress where
  sizeOf    _ = #{size    OxBitNetLoadProgress}
  alignment _ = #{alignment OxBitNetLoadProgress}
  peek p = CLoadProgress
    <$> #{peek OxBitNetLoadProgress, phase}    p
    <*> #{peek OxBitNetLoadProgress, loaded}   p
    <*> #{peek OxBitNetLoadProgress, total}    p
    <*> #{peek OxBitNetLoadProgress, fraction} p
  poke p v = do
    #{poke OxBitNetLoadProgress, phase}    p (clpPhase v)
    #{poke OxBitNetLoadProgress, loaded}   p (clpLoaded v)
    #{poke OxBitNetLoadProgress, total}    p (clpTotal v)
    #{poke OxBitNetLoadProgress, fraction} p (clpFraction v)

-- | @OxBitNetLoadOptions@
data CLoadOptions = CLoadOptions
  { cloOnProgress      :: !(FunPtr ProgressFn)
  , cloProgressUserdata :: !(Ptr ())
  , cloCacheDir         :: !(Ptr CChar)
  } deriving (Show)

instance Storable CLoadOptions where
  sizeOf    _ = #{size    OxBitNetLoadOptions}
  alignment _ = #{alignment OxBitNetLoadOptions}
  peek p = CLoadOptions
    <$> #{peek OxBitNetLoadOptions, on_progress}       p
    <*> #{peek OxBitNetLoadOptions, progress_userdata} p
    <*> #{peek OxBitNetLoadOptions, cache_dir}         p
  poke p v = do
    #{poke OxBitNetLoadOptions, on_progress}       p (cloOnProgress v)
    #{poke OxBitNetLoadOptions, progress_userdata} p (cloProgressUserdata v)
    #{poke OxBitNetLoadOptions, cache_dir}         p (cloCacheDir v)

-- | @OxBitNetChatMessage@
data CChatMessage = CChatMessage
  { ccmRole    :: !(Ptr CChar)
  , ccmContent :: !(Ptr CChar)
  } deriving (Show)

instance Storable CChatMessage where
  sizeOf    _ = #{size    OxBitNetChatMessage}
  alignment _ = #{alignment OxBitNetChatMessage}
  peek p = CChatMessage
    <$> #{peek OxBitNetChatMessage, role}    p
    <*> #{peek OxBitNetChatMessage, content} p
  poke p v = do
    #{poke OxBitNetChatMessage, role}    p (ccmRole v)
    #{poke OxBitNetChatMessage, content} p (ccmContent v)

-- --------------------------------------------------------------------
-- Callback types
-- --------------------------------------------------------------------

-- | Token callback: @int32_t (*)(const char *token, uintptr_t len, void *userdata)@
type TokenFn = Ptr CChar -> #{type uintptr_t} -> Ptr () -> IO Int32

-- | Progress callback: @void (*)(const OxBitNetLoadProgress *progress, void *userdata)@
type ProgressFn = Ptr CLoadProgress -> Ptr () -> IO ()

-- | Log callback: @void (*)(OxBitNetLogLevel level, const char *message, uintptr_t len, void *userdata)@
type LogFn = Word8 -> Ptr CChar -> #{type uintptr_t} -> Ptr () -> IO ()

foreign import ccall "wrapper"
  mkTokenFn :: TokenFn -> IO (FunPtr TokenFn)

foreign import ccall "wrapper"
  mkProgressFn :: ProgressFn -> IO (FunPtr ProgressFn)

foreign import ccall "wrapper"
  mkLogFn :: LogFn -> IO (FunPtr LogFn)

-- --------------------------------------------------------------------
-- FFI imports
-- --------------------------------------------------------------------

-- safe: long-running / callbacks
foreign import ccall safe "oxbitnet_load"
  oxbitnet_load :: Ptr CChar -> Ptr CLoadOptions -> IO (Ptr OxBitNet)

foreign import ccall safe "oxbitnet_generate"
  oxbitnet_generate :: Ptr OxBitNet -> Ptr CChar -> Ptr CGenerateOptions
                    -> FunPtr TokenFn -> Ptr () -> IO Int32

foreign import ccall safe "oxbitnet_chat"
  oxbitnet_chat :: Ptr OxBitNet -> Ptr CChatMessage -> #{type uintptr_t}
               -> Ptr CGenerateOptions -> FunPtr TokenFn -> Ptr () -> IO Int32

foreign import ccall safe "oxbitnet_free"
  oxbitnet_free :: Ptr OxBitNet -> IO ()

-- unsafe: quick / no callbacks
foreign import ccall unsafe "oxbitnet_default_generate_options"
  oxbitnet_default_generate_options :: IO CGenerateOptions

foreign import ccall unsafe "oxbitnet_default_load_options"
  oxbitnet_default_load_options :: IO CLoadOptions

foreign import ccall unsafe "oxbitnet_error_message"
  oxbitnet_error_message :: IO (Ptr CChar)

foreign import ccall safe "oxbitnet_set_logger"
  oxbitnet_set_logger :: FunPtr LogFn -> Ptr () -> Word8 -> IO ()
