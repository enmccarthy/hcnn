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
train 1 model trainSet trainKey = 
    let 
        vectorExample = (getX trainSet 1)
        vectorAns     = (getY trainKey 1)
        (vec, err)    = (forwardprop model (vectorExample, [])) in
    (backwardspropV2 (reverse model) (0.002) vectorAns err)

train iter model trainSet trainKey =
    let 
        vectorExample = (getX trainSet iter)
        vectorAns     = (getY trainKey iter)
        (vec, err)    = (forwardprop model (vectorExample, []))
        newMod        = (backwardspropV2 (reverse model) (0.002) vectorAns err) in
    -- must reverse becasue backprop is backwards
    train (iter-1) (reverse newMod) trainSet trainKey

-- test :: Int -> [([Float], [[Float]])] -> BS.ByteString -> BS.ByteString -> IO([([Float], [Integer])]) 
test 0 model testSet testKey = do
    n <- (`mod` 1000) <$> randomIO
    let vectorExample = getX testSet n
    let vectorAns = getY testKey n 
    let (vector, error) = (forwardprop model (vectorExample, []))
    return ([(vector, vectorAns)])
    -- let newtotal = (total+1)

    -- if ((elemIndex (maximum vector) vector) == (elemIndex (maximum vectorAns) vectorAns))
    --     then return ((correct+1)/newtotal)
    --     else return (correct/newtotal)
    
test iter model testSet testKey = do
    n <- (`mod` 1000) <$> randomIO
    let vectorExample = getX testSet n
    let vectorAns = getY testKey n
    let (vector, error) = (forwardprop model (vectorExample, []))

    restofTest <- (test (iter-1) model testSet testKey)
    --returning the vector output and the answer
    return ((vector, vectorAns):restofTest)
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
computeTotal :: [([Double],[Double])] -> Double
computeTotal [] = 0.0
computeTotal ((guess, actual):xs) = if (elemIndex (maximum guess) guess) == (elemIndex (maximum actual) actual) 
                                    then
                                        1 + (computeTotal xs)
                                    else
                                        (computeTotal xs)

render n = let s = " .:oO@" in s !! (fromIntegral n * length s `div` 256)

main :: IO ()
main = do
    -- almost this whole main came from https://crypto.stanford.edu/~blynn/haskell/brain.html 
    -- I wrote my own functions/main that test and train the neural net (what is left of it is above and in the comments below) 
    --but this is 11/10 prettier and way easier for debugging 
    [trainI, trainL, testI, testL] <- mapM ((decompress  <$>) . BS.readFile) [ "./src/train-images-idx3-ubyte.gz"
        , "./src/train-labels-idx1-ubyte.gz"
        ,  "./src/t10k-images-idx3-ubyte.gz"
        ,  "./src/t10k-labels-idx1-ubyte.gz"
        ]
    -- create model and remove the first layer because it is not necessary --
    let (m:model) = (createMod [(Input 784), (Hidden 30), (Output 10)] [(0, ((return [0.01]), (return [])))])

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

    putStrLn $ show (sum $ fromEnum <$> zipWith (==) guesses answers) ++
        " / 10000"
   -- train the model --
    let 
        -- vectorExample = getX trainI 1
        -- vectorAns = getY trainL 2
        -- vectorExample2 = getX trainI 2
        -- vectorExample3 = getX trainI 3
        -- (vec, err) = (forwardprop b2 (vectorExample, []))
        -- tmod = (backwardsprop (reverse b2) (0.002) vectorAns err)
    --     trainedMod = (train 1000 b2 trainI trainL)
    -- ansList <- (test 1000 (reverse trainedMod) testI testL)
    -- let tot = computeTotal ansList
    -- print (tot/1000.0) 
        

    return ()
