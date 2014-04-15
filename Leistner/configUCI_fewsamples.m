%initialize SSRF
%takes as input Number of subjects for train (labeled) and test (unlabeled)
%from UCI dataset
%test subjects are drawn from pool of test dataset

%in this version, Nsamp is the number of samples from each class

function forest = configUCI_fewsamples(Nsamp)

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

X = load('../Data/UCIHARDataset/train/X_train.txt');
Y = load('../Data/UCIHARDataset/train/y_train.txt');

subjects = load('../Data/UCIHARDataset/train/subject_train.txt');
subject_codes = unique(subjects);

%choosing the first training subject -- this is an example
subj = subject_codes(1); 
ind = find(subjects==subj);
X = X(ind, :);
Y = Y(ind, :);

classes = unique(Y);         %the class codes
n_class = length(classes); %the # of classes

% rng('default')   %fix random number generator seed

ind_sample = []; p = 0.01;  %take samples from 25% and 75% 

for i=1:n_class,
    ind_class = find(Y==classes(i));
    
    %finding the middle sample in each trial
    trans_end = find(diff(ind_class)>1);
    trans_start = trans_end+1;
    trans_start = [ind_class(1), ind_class(trans_start)'];
    trans_end = [ind_class(trans_end)', ind_class(end)];
    ind_middle = round((trans_start + trans_end)/2);
    %     ind_sample = [ind_sample ind_middle];
    
    for n = 1:Nsamp/2
        l1 = round(2*p*(trans_end(1)-ind_middle(1))); %there are 2 repetitions of each class (activity)
        l2 = round(p*0.5*(trans_end(2)-ind_middle(2)));
        ind_sample  = [ind_sample randi([ind_middle(1)-l1, ind_middle(1)+l1],1) randi([ind_middle(2)-l2, ind_middle(2)+l2],1)]
    end
            
 %     ind_sample = [ind_sample randsample(ind_class, Nsamp)'];

end
Xl = X(ind_sample,:);
Yl = Y(ind_sample);

inds = 1:size(X,1);
ind_nosample = inds(~ismember(inds, ind_sample));

Xu = X(ind_nosample, :);
Yu = Y(ind_nosample);

% repeating training data to balance the total number of training vs test
Xl = repmat(Xl, round(length(Yu)/length(Yl)), 1);
Yl = repmat(Yl, round(length(Yu)/length(Yl)), 1);




%% Params to init forests

PARAM = {ntrees,T0,alpha,tau,Xl,Yl,Xu,Yu,n_class};

forest = ssforest(PARAM);

end