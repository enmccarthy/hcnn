
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

    describe "softmax" $ do
        it "softmax calculation compared to scikitlearn" $
            softmax (DV.fromList [1.0,2.9,3.0]) `shouldBe`
              (DV.fromList [0.0663352,  0.4435102,  0.49015456])
