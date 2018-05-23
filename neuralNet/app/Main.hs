module Main where

import Codec.Compression.GZip (decompress)
import Lib
import qualified Data.ByteString.Lazy as BS
import Data.Functor
import Data.List
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

train 1 model trainI trainL = do
    n <- (`mod` 10000) <$> randomIO
    let vectorExample = getX trainI n
    let vectorAns = getY trainL n
    (vec, err) <- (forwardprop model (vectorExample, []))
    newMod <- (backwardsprop (reverse model) vec vectorAns (reverse err))
    return (reverse newMod)
train iter model trainI trainL = do
    n <- (`mod` 10000) <$> randomIO
    let vectorExample = getX trainI n
    let vectorAns = getY trainL n
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

    let (m:model) = (createMod [(Input 784), (Hidden 200), (Output 10)] [(0, ((return 0.01), (return [])))])
    -- let testmod = [(2, (1.0), return [[4.0, 3.0], [4.0, 3.0]])), (2, ((return 1.0), return [[2.0, 3.0], [2.0, 3.0]]))]
    trainedMod <- (train 100 model trainI trainL)
    
    let vectorExample = getX testI 5

    let vectorAns = getY testL 5


    let testVector = [2.0, 1.0]
    (vec, err) <- (forwardprop trainedMod (vectorExample, []))
    (print vec)
    -- (print err)


    --(vector2, err) <- (forwardprop model (vectorExample, []))
    (vector, error) <- (forwardprop model (vectorExample, []))
    -- (print vector2)
    (print vector) 
    -- (print (elemIndex (max vector) vector))
    --(print err)
    -- (print (cel [vector] vectorAns))
    -- (print (elemIndex (max vectorAns) vectorAns))
    return ()
