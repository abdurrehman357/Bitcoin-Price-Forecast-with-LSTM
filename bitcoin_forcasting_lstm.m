clc
clear all 
close all
 %load_dataSet
data=xlsread('BTC-Mints.csv');
%data=Monthly_BTC;
    dt=[data(:)];
 %display_data
 figure,
 plot(data)
 xlabel("Minuts")
 ylabel("Rates")
 title("Highest Rate of Bitcoin")
 
 %partition_of_data
 
 trec=numel(data);
 trrec=0.9*trec;
 NTST=floor(trrec);
 
 traindata=data(1:NTST+1);
 testdata=data(NTST+1:end);
 
 %Standrize_Data
mu=mean(traindata);
sig=std(traindata);
traindatastd=(traindata-mu)/sig;

%  figure,
%  plot(traindatastd)
%  xlabel("Weeks")
%  ylabel("Case")
%  title("Weekly Rate of BTC During Traing Phase")
%  
%  figure,
%  plot(data)
%  xlabel("Weeks")
%  ylabel("Case")
%  title("Weekly Rate of BTC During Traing Phase after Standrize")

%perdiction during Training Phase

Xtrain=traindatastd(1:end-1);
Ytrain=traindatastd(2:end);


%%Define LSTM  Network Arthcture
NoF=1;
NoR=1;
NHU=200;
layers=[
  sequenceInputLayer(NoF, 'Name' ,'ip')
  lstmLayer(NHU, 'Name','lstm')
  fullyConnectedLayer(NoR,'Name', 'FC')
  regressionLayer('Name','RL') ];

% lgraph=layerGraph(layers);
% plot(lgraph)

%Speicify the Training Options'

options=trainingOptions('adam','MaxEpochs',100,'GradientThreshold',1,'InitialLearnRate',0.005,'LearnRateSchedule','piecewise','LearnRateDropPeriod',125,'LearnRateDropFactor',0.2,'Verbose',0,'Plots','training-Progress');

%Train LSTM NEtwork
net=trainNetwork(Xtrain,Ytrain,layers,options);

%validtaion


dataTeststd=(testdata-mu)/sig;
Xtest=dataTeststd(1:end-1);
YTest=dataTeststd(2:end);

net=predictAndUpdateState(net,Xtrain);
[net, Ypred]=predictAndUpdateState(net,Ytrain(end));
NTSTs= numel(Xtest);
for i=2:NTSTs
    [net, Ypred(:,i)]=predictAndUpdateState(net,Ytrain(:,i-1));
end


%Unstand
Ypred=sig*Ypred+mu;
rmse=sqrt(mean(Ypred-YTest).^2);
ase=mean(mode(Ypred-YTest));
AMSE=mode(mean((Ypred-YTest).^2));

it=1:length(Ypred);
figure,
plot(traindata(1:end-1))
hold on
startpnt= NTST;
endpt= NTST+NTSTs;
idx=startpnt:endpt;
plot(idx,[data(startpnt) Ypred],'.-')
hold off

xlabel("Minuts")
ylabel("Rate")
title("Hourly ForeCast")
legend(["observed" "Forcast"])

figure,
subplot(2,1,1)
plot(testdata)
hold on
plot(Ypred,"-")
hold off
legend(["observed" "Forcast"])
subplot(2,1,2)
stem(Ypred-YTest)
ylabel("Error")
title("RMSE-"+rmse)
net=resetState(net);
net=predictAndUpdateState(net,Xtrain);
Ypred=[];
for i=1:NTSTs
    [net,Ypred(:,1)]=predictAndUpdateState(net,YTest(:,1));
end
Ypred-sig*Ypred + mu;
rmse=sqrt(mean(Ypred-YTest).^2);  
 

figure
subplot(2,1,1)
plot(testdata)
hold on
plot(Ypred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Rates")
xlabel("Minuts")
title("Forecast with updates")

subplot(2,1,1)
stem(Ypred-YTest)
ylabel("Error")
xlabel("Minuts")
title("RMSE-"+rmse)