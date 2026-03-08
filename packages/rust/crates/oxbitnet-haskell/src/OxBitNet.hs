{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}

-- | High-level, managed Haskell API for oxbitnet.
--
-- Typical usage:
--
-- @
-- 'withBitNet' "model.gguf" 'defaultLoadOptions' $ \\model -> do
--     'generate' model "Hello!" 'defaultGenerateOptions' $ \\token -> do
--         putStr token
--         return False  -- False = continue
-- @
module OxBitNet
  ( -- * Model handle
    BitNet
  , withBitNet
  , loadBitNet
  , freeBitNet

    -- * Generation
  , generate
  , chat

    -- * Logger
  , setLogger

    -- * Options
  , GenerateOptions(..)
  , defaultGenerateOptions
  , LoadOptions(..)
  , defaultLoadOptions

    -- * Types
  , ChatMessage(..)
  , userMessage
  , assistantMessage
  , systemMessage
  , LoadProgress(..)
  , LoadPhase(..)
  , LogLevel(..)

    -- * Exceptions
  , OxBitNetException(..)
  ) where

import Control.Concurrent.MVar
import Control.Exception (Exception, bracket, throwIO)
import Control.Monad (when)
import Data.IORef
import Data.Word (Word8, Word64)
import Foreign hiding (with)
import Foreign.C
import GHC.Generics (Generic)

import qualified Foreign.OxBitNet.Raw as Raw

-- --------------------------------------------------------------------
-- Types
-- --------------------------------------------------------------------

-- | Thread-safe handle to a loaded BitNet model.
newtype BitNet = BitNet (MVar (Maybe (Ptr Raw.OxBitNet)))

-- | Generation options.
data GenerateOptions = GenerateOptions
  { maxTokens     :: !Int    -- ^ Maximum number of tokens to generate (default: 256).
  , temperature   :: !Float  -- ^ Sampling temperature (default: 1.0).
  , topK          :: !Int    -- ^ Top-k sampling parameter (default: 50).
  , repeatPenalty :: !Float  -- ^ Repetition penalty (default: 1.1).
  , repeatLastN   :: !Int    -- ^ Window size for repetition penalty (default: 64).
  } deriving (Show, Eq)

-- | Sensible defaults for generation.
defaultGenerateOptions :: GenerateOptions
defaultGenerateOptions = GenerateOptions
  { maxTokens     = 256
  , temperature   = 1.0
  , topK          = 50
  , repeatPenalty = 1.1
  , repeatLastN   = 64
  }

-- | Options for loading a model.
data LoadOptions = LoadOptions
  { onProgress :: Maybe (LoadProgress -> IO ())
    -- ^ Progress callback (optional).
  , cacheDir   :: Maybe String
    -- ^ Cache directory path (optional).
  }

-- | Default load options (no progress callback, no cache dir).
defaultLoadOptions :: LoadOptions
defaultLoadOptions = LoadOptions Nothing Nothing

-- | Load progress information.
data LoadProgress = LoadProgress
  { lpPhase    :: !LoadPhase
  , lpLoaded   :: !Word64
  , lpTotal    :: !Word64
  , lpFraction :: !Double
  } deriving (Show)

-- | Load progress phase.
data LoadPhase
  = Download
  | Parse
  | Upload
  deriving (Show, Eq, Ord, Enum, Bounded)

-- | Log severity level.
data LogLevel
  = Trace
  | Debug
  | Info
  | Warn
  | Error
  deriving (Show, Eq, Ord, Enum, Bounded)

-- | A chat message with role and content.
data ChatMessage = ChatMessage
  { cmRole    :: !String
  , cmContent :: !String
  } deriving (Show, Eq)

-- | Construct a user message.
userMessage :: String -> ChatMessage
userMessage = ChatMessage "user"

-- | Construct an assistant message.
assistantMessage :: String -> ChatMessage
assistantMessage = ChatMessage "assistant"

-- | Construct a system message.
systemMessage :: String -> ChatMessage
systemMessage = ChatMessage "system"

-- | Exceptions thrown by oxbitnet operations.
data OxBitNetException
  = LoadError String
  | GenerateError String
  | Disposed
  deriving (Show, Eq, Generic, Exception)

-- --------------------------------------------------------------------
-- Model lifecycle
-- --------------------------------------------------------------------

-- | Load a model, run an action, then free it. Ensures cleanup on exception.
withBitNet :: String -> LoadOptions -> (BitNet -> IO a) -> IO a
withBitNet source opts = bracket (loadBitNet source opts) freeBitNet

-- | Load a model from a URL or local file path.
loadBitNet :: String -> LoadOptions -> IO BitNet
loadBitNet source opts =
  withCString source $ \cSource -> do
    cOpts <- Raw.oxbitnet_default_load_options

    -- Set up progress callback
    progressFunPtr <- case onProgress opts of
      Nothing -> return nullFunPtr
      Just cb -> Raw.mkProgressFn $ \pProgress _userdata -> do
        raw <- peek pProgress
        let progress = LoadProgress
              { lpPhase    = toLoadPhase (Raw.clpPhase raw)
              , lpLoaded   = Raw.clpLoaded raw
              , lpTotal    = Raw.clpTotal raw
              , lpFraction = realToFrac (Raw.clpFraction raw)
              }
        cb progress

    let cOpts' = cOpts
          { Raw.cloOnProgress = progressFunPtr
          }

    -- Set up cache dir
    result <- case cacheDir opts of
      Nothing -> alloca $ \pOpts -> do
        poke pOpts cOpts'
        Raw.oxbitnet_load cSource pOpts
      Just dir -> withCString dir $ \cDir -> alloca $ \pOpts -> do
        poke pOpts cOpts' { Raw.cloCacheDir = cDir }
        Raw.oxbitnet_load cSource pOpts

    -- Free progress FunPtr (no longer needed after load completes)
    when (progressFunPtr /= nullFunPtr) $
      freeHaskellFunPtr progressFunPtr

    if result == nullPtr
      then do
        errPtr <- Raw.oxbitnet_error_message
        msg <- if errPtr == nullPtr
               then return "Failed to load model"
               else peekCString errPtr
        throwIO (LoadError msg)
      else BitNet <$> newMVar (Just result)

-- | Free a model handle. Safe to call multiple times.
freeBitNet :: BitNet -> IO ()
freeBitNet (BitNet mvar) = modifyMVar_ mvar $ \mHandle -> do
  case mHandle of
    Nothing  -> return Nothing
    Just ptr -> do
      Raw.oxbitnet_free ptr
      return Nothing

-- | Acquire the raw pointer, throwing 'Disposed' if already freed.
withHandle :: BitNet -> (Ptr Raw.OxBitNet -> IO a) -> IO a
withHandle (BitNet mvar) action = withMVar mvar $ \mHandle ->
  case mHandle of
    Nothing  -> throwIO Disposed
    Just ptr -> action ptr

-- --------------------------------------------------------------------
-- Generation
-- --------------------------------------------------------------------

-- | Generate text from a raw prompt.
--
-- The callback receives each token as a 'String'. Return 'True' to stop
-- generation early, 'False' to continue.
generate :: BitNet -> String -> GenerateOptions -> (String -> IO Bool) -> IO ()
generate model prompt opts onToken =
  withHandle model $ \handle ->
  withCString prompt $ \cPrompt ->
  withCGenerateOptions opts $ \pOpts -> do
    tokenFn <- Raw.mkTokenFn $ \cStr len _userdata -> do
      str <- peekCStringLen (cStr, fromIntegral len)
      stop <- onToken str
      return (if stop then 1 else 0)
    ret <- Raw.oxbitnet_generate handle cPrompt pOpts tokenFn nullPtr
    freeHaskellFunPtr tokenFn
    when (ret /= 0) $ do
      errPtr <- Raw.oxbitnet_error_message
      msg <- if errPtr == nullPtr
             then return "Generate failed"
             else peekCString errPtr
      throwIO (GenerateError msg)

-- | Generate text from chat messages.
--
-- The callback receives each token as a 'String'. Return 'True' to stop
-- generation early, 'False' to continue.
chat :: BitNet -> [ChatMessage] -> GenerateOptions -> (String -> IO Bool) -> IO ()
chat model messages opts onToken =
  withHandle model $ \handle ->
  withCChatMessages messages $ \pMsgs numMsgs ->
  withCGenerateOptions opts $ \pOpts -> do
    tokenFn <- Raw.mkTokenFn $ \cStr len _userdata -> do
      str <- peekCStringLen (cStr, fromIntegral len)
      stop <- onToken str
      return (if stop then 1 else 0)
    ret <- Raw.oxbitnet_chat handle pMsgs (fromIntegral numMsgs) pOpts tokenFn nullPtr
    freeHaskellFunPtr tokenFn
    when (ret /= 0) $ do
      errPtr <- Raw.oxbitnet_error_message
      msg <- if errPtr == nullPtr
             then return "Chat failed"
             else peekCString errPtr
      throwIO (GenerateError msg)

-- --------------------------------------------------------------------
-- Logger
-- --------------------------------------------------------------------

-- | Install a global logger. Must be called before loading any model.
-- Can only be called once; subsequent calls are no-ops.
--
-- The 'FunPtr' is intentionally leaked (process-lifetime, matching the C API
-- contract that the logger is installed once and lives forever).
setLogger :: LogLevel -> (LogLevel -> String -> IO ()) -> IO ()
setLogger minLevel cb = do
  funPtr <- Raw.mkLogFn $ \level cStr len _userdata -> do
    str <- peekCStringLen (cStr, fromIntegral len)
    cb (toLogLevel level) str
  Raw.oxbitnet_set_logger funPtr nullPtr (fromIntegral (fromEnum minLevel))

-- --------------------------------------------------------------------
-- Internal helpers
-- --------------------------------------------------------------------

toLoadPhase :: Raw.LoadPhase -> LoadPhase
toLoadPhase p
  | p == Raw.PhaseDownload = Download
  | p == Raw.PhaseParse    = Parse
  | p == Raw.PhaseUpload   = Upload
  | otherwise              = Download  -- fallback

toLogLevel :: Word8 -> LogLevel
toLogLevel n
  | n <= 0    = Trace
  | n == 1    = Debug
  | n == 2    = Info
  | n == 3    = Warn
  | otherwise = Error

withCGenerateOptions :: GenerateOptions -> (Ptr Raw.CGenerateOptions -> IO a) -> IO a
withCGenerateOptions opts action = alloca $ \p -> do
  poke p Raw.CGenerateOptions
    { Raw.cgoMaxTokens     = fromIntegral (maxTokens opts)
    , Raw.cgoTemperature   = realToFrac (temperature opts)
    , Raw.cgoTopK          = fromIntegral (topK opts)
    , Raw.cgoRepeatPenalty = realToFrac (repeatPenalty opts)
    , Raw.cgoRepeatLastN   = fromIntegral (repeatLastN opts)
    }
  action p

-- | Marshal a list of 'ChatMessage' into a C array, using nested
-- 'withCString' to keep all strings alive for the duration of the action.
withCChatMessages :: [ChatMessage] -> (Ptr Raw.CChatMessage -> Int -> IO a) -> IO a
withCChatMessages msgs action = go msgs [] where
  go [] acc = do
    let n = length acc
        cMsgs = reverse acc
    allocaArray n $ \pArr -> do
      pokeArray pArr cMsgs
      action pArr n
  go (ChatMessage role content : rest) acc =
    withCString role $ \cRole ->
    withCString content $ \cContent ->
      go rest (Raw.CChatMessage cRole cContent : acc)
