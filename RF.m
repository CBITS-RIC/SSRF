clear all 
close all

%%LOAD Train Data
accx = load ('../Data/UCIHARDataset/train/InertialSignals/total_acc_x_train.txt'); 
accy = load('../Data/UCIHARDataset/train/InertialSignals/total_acc_y_train.txt');   
accz = load('../Data/UCIHARDataset/train/InertialSignals/total_acc_z_train.txt');   

codesTrue = load('../Data/UCIHARDataset/train/y_train.txt');

%Store Code and label of each unique State
% StateCodes = cell(length(uniqStates),2);
% StateCodes(:,1) = 
% StateCodes(:,2) = unique(labels); %sorted by unique


%extract features
features = [];
disp('extracting features')
for k = 1:size(accx,1)
    
    acc = [accx(k,:);accy(k,:);accz(k,:)];
    [fvec flab] = getFeatures(acc);
    features = [features;fvec;]; 
end
disp('Features extracted')

%Scale features ([0,1]
featureMax = max(features,[],1);
featureMin = min(features,[],1);
features = (features - repmat(featureMin,size(features,1),1))*spdiags(1./(featureMax-featureMin)',0,size(features,2),size(features,2));

%Train RF
disp('RF Train');

ntrees = 400;
OOBVarImp = 'off';   %enable variable importance measurement

forest = TreeBagger(ntrees,features,codesTrue','OOBVarImp',OOBVarImp);

%plot oobe
if strcmp(OOBVarImp,'on')
    figure
    plot(oobError(forest));
    
    %plot var importance
    if strcmp(OOBVarImp,'on')
        figure('name',['k-fold ' num2str(k)])
        barh(forest.OOBPermutedVarDeltaError);
        set(gca,'ytick',1:length(trainingClassifierData.featureLabels),'yticklabel',trainingClassifierData.featureLabels)
    end
end
%% Test RF

%Load Test data
accx = load ('../Data/UCIHARDataset/test/InertialSignals/total_acc_x_test.txt');  
accy = load ('../Data/UCIHARDataset/test/InertialSignals/total_acc_y_test.txt');  
accz = load ('../Data/UCIHARDataset/test/InertialSignals/total_acc_z_test.txt');  

codesTrue = load('../Data/UCIHARDataset/test/y_test.txt');

%extract features
features = [];
disp('extracting features')
for k = 1:size(accx,1)
    
    acc = [accx(k,:);accy(k,:);accz(k,:)];
    [fvec flab] = getFeatures(acc);
    features = [features;fvec;]; 
end
disp('Features extracted')

%Scale features ([0,1]
featureMax = max(features,[],1);
featureMin = min(features,[],1);
features = (features - repmat(featureMin,size(features,1),1))*spdiags(1./(featureMax-featureMin)',0,size(features,2),size(features,2));

%Predict 
[codesRF,P_RF] = predict(forest,features);

%results
[matRF,accRF] = createConfusionMatrix(codesTrue,codesRF);
disp(['accRF = ',accRF])
figure
imagesc(matRF), colormap gray
figure, hold on
plot(codesTrue,'g'); plot(cell2vec(codesRF),'r')




