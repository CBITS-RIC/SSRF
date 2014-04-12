%initialize SSRF
%takes as input Number of subjects for train (labeled) and test (unlabeled)
%from UCI dataset
%test subjects are drawn from pool of test dataset

function forest = configUCI(Ntr,Nte)

%parameters
ntrees = 10;     %forest size
T0 = 5;          %initial temperature
alpha = 1;     %coeff to control the weight of the unlabeled part in the loss function
tau = 60;  %cooling fcn time constant


%% g50c dataset
% load g50c.mat;
% Y = y/2+1.5;
% n_class = length(unique(Y)); %the classes
% Xl = X(idxLabs(1,:),:);    Yl = Y(idxLabs(1,:));
% Xu = X(idxUnls(1,:),:);    Yu = Y(idxUnls(1,:));

%% UCI Dataset

% frac = 0.1;    %the frac of labeled data


% disp('Load UCI Dataset')

Xtrain = load('../Data/UCIHARDataset/train/X_train.txt');
Ytrain = load('../Data/UCIHARDataset/train/y_train.txt');
Xtest = load('../Data/UCIHARDataset/test/X_test.txt');   
Ytest = load('../Data/UCIHARDataset/test/y_test.txt');   

%Randomly selects a portion of training data as labeled. Rest is Unlabeled
% labeled = zeros(size(Xtrain,1),1);
% labeled(randperm(size(Xtrain,1),round(size(Xtrain,1)*frac))) = 1;
% labeled = logical(labeled);
% Xl = Xtrain(labeled,:);   Yl = Ytrain(labeled);          %labeled data
% Xu = Xtrain(~labeled,:);  Yu = Ytrain(~labeled);   %unlabeled data

%Select # of subjects for labeled (train) and unlabeled (test) 
subjects_train = load('../Data/UCIHARDataset/train/subject_train.txt');
subjects_test = load('../Data/UCIHARDataset/test/subject_test.txt');
subject_codes_train = unique(subjects_train);
subject_codes_test = unique(subjects_test);

%randomly select subject
% rng('default')   %fix random number generator seed
% 
% ind = randperm(length(unique(subjects)));
% subj_train = ind(1:n_train);
% subj_test  = ind(n_train+1:n_train+n_test);

%ordered selection of subjects
subj_train = subject_codes_train(1:Ntr);
subj_test  = subject_codes_test(1:Nte);

ind_train = [];
for i=1:Ntr,
    ind_train = [ind_train; find(subjects_train==subj_train(i))];
end
ind_test = [];
for i=1:Nte,
    ind_test = [ind_test; find(subjects_test==subj_test(i))];
end

n_class = length(unique(Ytrain)); %the # of classes
% classes = unique(Ytrain);         %the class codes


Xl = Xtrain(ind_train,:);   Yl = Ytrain(ind_train);          %labeled data
% Xl = repmat(Xtrain(ind_train,:),3,1);   Yl = repmat(Ytrain(ind_train),3,1);          %labeled data
Xu = Xtest(ind_test,:);  Yu = Ytest(ind_test);   %unlabeled data
% Yu_forest = zeros(ntrees*size(Xu,1),1);          %unlabeled data for entire forest

%% Params to init forests

PARAM = {ntrees,T0,alpha,tau,Xl,Yl,Xu,Yu,n_class};

forest = ssforest(PARAM);

end