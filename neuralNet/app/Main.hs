module Main where

import Codec.Compression.GZip (decompress)
import Lib
import qualified Data.ByteString.Lazy as BS
import Data.Functor
import Data.List
import Data.Ord
import Control.Monad
-- import qualified Data.Vector as DV
import System.Random
    


createMod :: InputLayers -> Model -> Model
createMod [] mod = mod
createMod ip mod = let (currMod, currIp) = (initModel mod ip) in
    (createMod currIp currMod)

-- These came from https://crypto.stanford.edu/~blynn/haskell/brain.html
getImage s n = fromIntegral . BS.index s . (n*28^2 + 16 +) <$> [0..28^2 - 1]
getX     s n = (/ 256) <$> getImage s n
getLabel s n = fromIntegral $ BS.index s (n + 8)
getY     s n = fromIntegral . fromEnum . (getLabel s n ==) <$> [0..9]

-- train the model with given test sets -- 
-- takes in a number of iterations a model and the training set and solutions
-- train :: Int -> [([Float], [[Float]])] -> BS.ByteString -> BS.ByteString -> [([Float], [[Float]])] 
-- train 1 model trainSet trainKey = 
--     let 
--         vectorExample = (getX trainSet 1)
--         vectorAns     = (getY trainKey 1)
--         (vec, err)    = (forwardprop model (vectorExample, [])) in
--     (backwardsprop (reverse model) vec vectorAns (reverse err))

-- train iter model trainSet trainKey =
--     let 
--         vectorExample = (getX trainSet iter)
--         vectorAns     = (getY trainKey iter)
--         (vec, err)    = (forwardprop model (vectorExample, []))
--         newMod        = (backwardsprop (reverse model) vec vectorAns (reverse err)) in
--     -- must reverse becasue backprop is backwards
--     train (iter-1) (reverse newMod) trainSet trainKey

-- test :: Int -> [([Float], [[Float]])] -> BS.ByteString -> BS.ByteString -> IO([([Float], [Integer])]) 
-- test 0 model testSet testKey = do
--     n <- (`mod` 1000) <$> randomIO
--     let vectorExample = getX testSet n
--     let vectorAns = getY testKey n 
--     let (vector, error) = (forwardprop model (vectorExample, []))
--     return ([(vector, vectorAns)])
--     -- let newtotal = (total+1)

--     -- if ((elemIndex (maximum vector) vector) == (elemIndex (maximum vectorAns) vectorAns))
--     --     then return ((correct+1)/newtotal)
--     --     else return (correct/newtotal)
    
-- test iter model testSet testKey = do
--     n <- (`mod` 1000) <$> randomIO
--     let vectorExample = getX testSet n
--     let vectorAns = getY testKey n
--     let (vector, error) = (forwardprop model (vectorExample, []))

--     restofTest <- (test (iter-1) model testSet testKey)
--     --returning the vector output and the answer
--     return((vector, vectorAns):restofTest)
    -- (print vectorAns)
    -- if ((elemIndex (maximum vector) vector) == (elemIndex (maximum vectorAns) vectorAns)) 
    --     then test (iter-1) model testSet testKey (correct+1) newtotal 
    --     else test (iter-1) model testSet testKey correct newtotal

-- convert model to not have the i, aka the number of previous nodes --
convertMod ((i, (b,w)):[]) = do
    bias <- b
    weight <- w
    return ([(bias, weight)])
convertMod ((i,(b,w)):rm) = do
    bias <- b
    weight <- w
    restMod <- (convertMod rm)
    return ((bias, weight):restMod)

render n = let s = " .:oO@" in s !! (fromIntegral n * length s `div` 256)

main :: IO ()
main = do
    -- these came from https://crypto.stanford.edu/~blynn/haskell/brain.html
    [trainI, trainL, testI, testL] <- mapM ((decompress  <$>) . BS.readFile) [ "./src/train-images-idx3-ubyte.gz"
        , "./src/train-labels-idx1-ubyte.gz"
        ,  "./src/t10k-images-idx3-ubyte.gz"
        ,  "./src/t10k-labels-idx1-ubyte.gz"
        ]
    -- create model and remove the first layer because it is not necessary --
    let (m:model) = (createMod [(Input 784), (Hidden 30), (Output 10)] [(0, ((return [0.01]), (return [])))])
    -- changed formats half way through -- 
    b <- (convertMod model)
    n <- (`mod` 10000) <$> randomIO
    putStr . unlines $
        take 28 $ take 28 <$> iterate (drop 28) (render <$> getImage testI n)

    let
        example = getX testI n
        bs = scanl (foldl' (\b n -> learn (getX trainI n) (getY trainL n) b)) b [
            [   0.. 999],
            [1000..2999],
            [3000..5999],
            [6000..9999]]
        smart = last bs
        cute d score = show d ++ ": " ++ replicate (round $ 70 * min 1 score) '+'
        bestOf = fst . maximumBy (comparing snd) . zip [0..]

    forM_ bs $ putStrLn . unlines . zipWith cute [0..9] . feed example

    putStrLn $ "best guess: " ++ show (bestOf $ feed example smart)

    let guesses = bestOf . (\n -> feed (getX testI n) smart) <$> [0..9999]
    let answers = getLabel testL <$> [0..9999]
    -- train the model --
    -- let vectorExample = getX trainI 1
    -- let vectorAns = getY trainL 2
    -- let vectorExample2 = getX trainI 2
    -- let vectorExample3 = getX trainI 3
    -- let (vec, err) = (forwardprop otherFormat (vectorExample, []))
    -- let (t:tmod) = (backwardsprop (reverse otherFormat) vec vectorAns (reverse err))
    -- let trainedMod = (train 100 otherFormat trainI trainL)
    --     newtrainedMod = trainedMod
    -- -- testing the model -- 
    -- let newtmod = (reverse tmod)
    -- print (t)
    -- print (vec)
    -- let (vec2, err2) = (forwardprop (newtmod) (vectorExample2, []))
    -- let (vec3, err3) = (forwardprop (reverse tmod) (vectorExample3, []))
    -- (print vec2)
    -- (print vec3)
    -- total <- (test 3 newtrainedMod trainI trainL)
    -- (print total)

    return ()
