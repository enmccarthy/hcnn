module Main where

import Codec.Compression.GZip (decompress)
import Lib
import qualified Data.ByteString.Lazy as BS
import Data.Functor
import qualified Data.Vector as DV
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

train 1 model trainI trainL = do
    n <- (`mod` 10000) <$> randomIO
    let example = getX trainI n
    let vectorExample = (DV.fromList example)
    let ans = getY trainL n
    let vectorAns = (DV.fromList ans)
    (vec, err) <- (forwardprop model (vectorExample, []))
    newMod <- (backwardsprop (reverse model) vec vectorAns (reverse err))
    return (reverse newMod)
train iter model trainI trainL = do
    n <- (`mod` 10000) <$> randomIO
    let example = getX trainI n
    let vectorExample = (DV.fromList example)
    let ans = getY trainL n
    let vectorAns = (DV.fromList ans)
    (vec, err) <- (forwardprop model (vectorExample, []))
    newMod <- (backwardsprop (reverse model) vec vectorAns (reverse err))
    train (iter-1) (reverse newMod) trainI trainL

main :: IO ()
main = do
    -- these came from https://crypto.stanford.edu/~blynn/haskell/brain.html
    [trainI, trainL, testI, testL] <- mapM ((decompress  <$>) . BS.readFile) [ "./src/train-images-idx3-ubyte.gz"
        , "./src/train-labels-idx1-ubyte.gz"
        ,  "./src/t10k-images-idx3-ubyte.gz"
        ,  "./src/t10k-labels-idx1-ubyte.gz"
        ]

    let (m:model) = (createMod [(Input 784), (Hidden 5), (Output 10)] [(0, ((return 0.01), (return [])))])

    trainedMod <- (train 1000 model trainI trainL)
    let example = getX trainI 2
    let vectorExample = (DV.fromList example)
    let ans = getY trainL 2
    let vectorAns = (DV.fromList ans)
    (vector2, err) <- (forwardprop model (vectorExample, []))
    (vector, err) <- (forwardprop trainedMod (vectorExample, []))
    (print vector2)
    (print vector)
    -- (print err)
    (print (cel [vector] vectorAns))
    (print vectorAns)
    return ()
