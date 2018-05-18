module Main where

import Lib
import qualified Data.Vector as DV


createMod :: InputLayers -> Model -> Model
createMod [] mod = mod
createMod ip mod = let (currMod, currIp) = (initModel mod ip) in
    (createMod currIp currMod)

main :: IO ()
main = do
    lis <- (createWHelp 3)
    (print lis)
    otherList <- (createWHelp 3) 
    let (m:model) = (createMod [(Input 3), (Hidden 3), (Output 2)] [(0, ((return 0.01), (return [])))])
    (vec, err) <- (forwardprop model (lis, []))
    (print vec)
    (print err)
    let revmodel = (reverse model)
    newMod <- (backwardsprop model vec otherList err) 
    testMod <- (forwardprop model (lis, []))
    (print testMod)
    return ()
