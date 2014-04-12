%%Train and Test RF on UCI dataset

clear all
close all

%%LOAD Train Data
% accx = load ('../Data/UCIHARDataset/train/InertialSignals/total_acc_x_train.txt');
% accy = load('../Data/UCIHARDataset/train/InertialSignals/total_acc_y_train.txt');
% accz = load('../Data/UCIHARDataset/train/InertialSignals/total_acc_z_train.txt');
%
% gyrx = load ('../Data/UCIHARDataset/train/InertialSignals/body_gyro_x_train.txt');
% gyry = load('../Data/UCIHARDataset/train/InertialSignals/body_gyro_y_train.txt');
% gyrz = load('../Data/UCIHARDataset/train/InertialSignals/body_gyro_z_train.txt');

codesTrue = load('../Data/UCIHARDataset/train/y_train.txt');
features = load('../Data/UCIHARDataset/train/X_train.txt');

uniqStates = unique(codesTrue);

%Store Code and label of each unique State
% StateCodes = cell(length(uniqStates),2);
% StateCodes(:,1) =
% StateCodes(:,2) = unique(labels); %sorted by unique


%extract features
% features_acc = []; features_gyr = []; features = [];
% disp('extracting features')
% for k = 1:size(accx,1)
%
%     acc = [accx(k,:);accy(k,:);accz(k,:)];
%     [fvec, flab_acc] = getFeatures(acc);
%     features_acc = [features_acc;fvec;];
%
%     gyr = [gyrx(k,:);gyry(k,:);gyrz(k,:)];
%     [fvec, flab_gyr] = getFeatures(gyr);
%     features_gyr = [features_gyr;fvec;];
%
%     features = [features_acc features_gyr];
%     flab = [flab_acc;flab_gyr];
% end
% disp('Features extracted')
%
% %Scale features ([0,1]
% featureMax = max(features,[],1);
% featureMin = min(features,[],1);
% features = (features - repmat(featureMin,size(features,1),1))*spdiags(1./(featureMax-featureMin)',0,size(features,2),size(features,2));


%Train RF
disp('RF Train');

ntrees = 30;
OOBVarImp = 'off';   %enable variable importance measurement

forest = TreeBagger(ntrees,features,codesTrue','OOBVarImp',OOBVarImp);

%one classification tree
% tree = classregtree(features,codesTrue,'method','classification');
% view(tree,'mode','graph') % graphic description

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

%% Init HMM




%compute emission prob matrix from P_RF on train data
[~,P_RF] = predict(forest,features);


%inialize parameters for hmm
d       = length(uniqStates);   %number of symbols (=#states)
nstates = d;                    %number of states
mu      = zeros(d,nstates);     %mean of emission distribution
sigma   = zeros(d,1,nstates);   %std dev of emission distribution
Pi      = ones(length(uniqStates),1) ./ length(uniqStates); %uniform prior
sigmaC  = .1;                   %use a constant std dev
PTrain = P_RF;   %Emission prob = RF class prob for training data
pnt = 0.9;             %prob of remaining in current state
A = eye(d)*pnt;        %the transition matrix 
A(A == 0) = (1-pnt)./(d-1);


%create emission probabilities for HMM
PBins  = cell(d,1);

%for each type of state we need a distribution
for bin = 1:d
    clipInd         = find(codesTrue==uniqStates(bin)); %find clip indices
    PBins{bin,1}    = PTrain(clipInd,:);    %The emission probability of the HMM is the RF prob over training data
    mu(:,bin)       = mean(PBins{bin,1}); %mean
    sigma(:,:,bin)  = sigmaC; %set std dev
end

%create distribution for pmtk3 package
emission        = struct('Sigma',[],'mu',[],'d',[]);
emission.Sigma  = sigma;
emission.mu     = mu;
emission.d      = d;

%construct HMM using pmtk3 package
HMMmodel           = hmmCreate('gauss',Pi,A,emission);
HMMmodel.emission  = condGaussCpdCreate(emission.mu,emission.Sigma);
HMMmodel.fitType   = 'gauss';



%% Test RF
disp('RF Test');

%Load Test data
% accx = load ('../Data/UCIHARDataset/test/InertialSignals/total_acc_x_test.txt');
% accy = load ('../Data/UCIHARDataset/test/InertialSignals/total_acc_y_test.txt');
% accz = load ('../Data/UCIHARDataset/test/InertialSignals/total_acc_z_test.txt');

codesTrue = load('../Data/UCIHARDataset/test/y_test.txt');
features = load('../Data/UCIHARDataset/test/X_test.txt');

%extract features
% features_acc = []; features_gyr = []; features = [];
% disp('extracting features')
% for k = 1:size(accx,1)
%
%      acc = [accx(k,:);accy(k,:);accz(k,:)];
%     [fvec, flab_acc] = getFeatures(acc);
%     features_acc = [features_acc;fvec;];
%
%     gyr = [gyrx(k,:);gyry(k,:);gyrz(k,:)];
%     [fvec, flab_gyr] = getFeatures(gyr);
%     features_gyr = [features_gyr;fvec;];
%
%     features = [features_acc features_gyr];
% end
% disp('Features extracted')

%Scale features ([0,1]
% featureMax = max(features,[],1);
% featureMin = min(features,[],1);
% features = (features - repmat(featureMin,size(features,1),1))*spdiags(1./(featureMax-featureMin)',0,size(features,2),size(features,2));

%Predict on test data
[codesRF,P_RF] = predict(forest,features);

%one classification tree
% codesRF = eval(tree,features);

%predict w HMM
PTest = P_RF;        %The observation sequence (Test data)
[gamma, ~, ~, ~, ~]   = hmmInferNodes(HMMmodel,PTest');
[maxP, codesHMM] = max(gamma);
codesHMM = codesHMM';


%results
codesRF = cell2vec(codesRF);
matRF = confusionmat(codesTrue,codesRF);
matHMM = confusionmat(codesTrue,codesHMM);
accRF = length(find(codesRF==codesTrue))/length(codesTrue);
accHMM = length(find(codesHMM==codesTrue))/length(codesTrue);

disp(['accRF = ' num2str(accRF)])
disp(['accHMM = ' num2str(accHMM)])

figure, subplot(121)
imagesc(matRF), colormap gray
subplot(122), imagesc(matHMM), colormap gray

figure, hold on
plot(codesTrue,'g'); plot(codesRF,'r'), plot(codesHMM,'b')




