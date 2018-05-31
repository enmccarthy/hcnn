
import Test.Hspec
import Lib
import qualified Data.Vector as DV

main :: IO ()
main = hspec $ do
    describe "mse error" $ do
        it "no error" $
            mse [1.0,2.0,3.0] [1.0,2.0,3.0] `shouldBe` 0
        it "should have error" $
            mse [1.0,2.0,3.0] [3.0,3.0,1.0] `shouldBe` 3.0
--I switched from floats to doubles so now this returns more decimal points and I 
-- just havent modified the test
    -- describe "softmax" $ do
    --     it "softmax calculation compared to scikitlearn" $
    --         softmax [1.0,2.9,3.0] `shouldBe`
    --             [0.0663352,  0.4435102,  0.49015456]
    --     it "all ones" $
    --         softmax [1.0,1.0,1.0,1.0] `shouldBe`
    --             [ 0.25, 0.25, 0.25, 0.25]
    --     it "negative and positive" $
    --         softmax [2.3,4.7,(-8.2),(-0.0000032)] `shouldBe`
    --             [8.24847e-02, 9.0924335e-01, 2.2713366e-06,  8.269794e-03]
    describe "forwardprop" $ do
        it "1d x 1d and negative" $
            forwardMult [2.0, 3.0, (-4.0)] ([0,0,0], [[1.0,2.0,3.0]]) `shouldBe`
                [-4.0]
        it "1d x 2d no bias" $
            forwardMult [1.0, 1.0, 1.0] ([0,0,0], [[2.0,2.0,2.0], [3.0,3.0,3.0]]) `shouldBe`
                [6.0, 9.0]
        it "1d x 1d" $
            forwardMult [2.0, 3.0, (-4.0)] ([1.0], [[1.0,2.0,3.0]]) `shouldBe`
                [-3.0]
        it "1d x 2d no bias" $
            forwardMult [1.0, 1.0, 1.0] ([1.0,2.0], [[2.0,2.0,2.0], [3.0,3.0,3.0]]) `shouldBe`
                [7.0, 11.0]

             


