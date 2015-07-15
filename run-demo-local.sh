./local-helper.sh MLbenchmark.driver \
--trainFile=data/small_train.dat \
--testFile=data/small_test.dat \
--numFeatures=9947 \
--numRounds=100 \
--localIterFrac=0.1 \
--numSplits=4 \
--lambda=0.002 \
--chkptDir=dir/ \
--justCoCoA=false
