module Main where

import Control.Monad (unless)
import Data.IORef
import System.Environment (getArgs)
import System.IO (hFlush, stdout, hIsEOF, stdin, hSetBuffering, BufferMode(..))

import OxBitNet

main :: IO ()
main = do
  hSetBuffering stdout NoBuffering
  args <- getArgs
  source <- case args of
    [s] -> return s
    _   -> error "Usage: oxbitnet-chat <model-path-or-url>"

  let loadOpts = defaultLoadOptions
        { onProgress = Just $ \p -> do
            let phaseName = case lpPhase p of
                  Download -> "Download"
                  Parse    -> "Parse"
                  Upload   -> "Upload"
                pct = lpFraction p * 100.0 :: Double
            putStr $ "\r[" ++ phaseName ++ "] " ++ showFFloat1 pct ++ "%"
            hFlush stdout
        }

  putStrLn "Loading model..."
  withBitNet source loadOpts $ \model -> do
    putStrLn "\nModel loaded. Type a message (Ctrl-D to quit).\n"
    history <- newIORef ([] :: [ChatMessage])
    chatLoop model history

chatLoop :: BitNet -> IORef [ChatMessage] -> IO ()
chatLoop model history = do
  putStr "> "
  hFlush stdout
  eof <- hIsEOF stdin
  unless eof $ do
    input <- getLine
    unless (null input) $ do
      modifyIORef' history (++ [userMessage input])
      msgs <- readIORef history

      responseRef <- newIORef ""
      chat model msgs defaultGenerateOptions $ \token -> do
        putStr token
        modifyIORef' responseRef (++ token)
        return False

      putStrLn ""
      response <- readIORef responseRef
      modifyIORef' history (++ [assistantMessage response])

    chatLoop model history

-- | Show a Double with 1 decimal place.
showFFloat1 :: Double -> String
showFFloat1 x =
  let n = round (x * 10) :: Int
      whole = n `div` 10
      frac = n `mod` 10
  in show whole ++ "." ++ show (abs frac)
