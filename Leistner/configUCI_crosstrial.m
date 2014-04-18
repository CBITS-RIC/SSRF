%initialize SSRF
%takes as input Number of subjects for train (labeled) and test (unlabeled)
%from UCI dataset
%test subjects are drawn from pool of test dataset

% Nsamp: Number of samples from each trial
% RepFac: Repetition Factor - a number between 0 and 1 (1=repeating labeled samples until they become
%         equal in amount to unlabeled samples; 0 = no repetition).

function forest = configUCI_crosstrial(trial)

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

% X = X(:, 1:80);%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subjects = load('../Data/UCIHARDataset/train/subject_train.txt');
subject_codes = unique(subjects);

%choosing the first training subject -- this is an example
subj = subject_codes(3);
ind = find(subjects==subj);
X = X(ind, :);
Y = Y(ind, :);

classes = unique(Y);         %the class codes
n_class = length(classes); %the # of classes

% rng('default')   %fix random number generator seed

ind_sample = []; p = 0.5;  %take samples from mid-p/2 to mid+p/2 (of the trial length)

for i=1:n_class,
    ind_class = find(Y==classes(i));

    %finding the transitions between trials
    trans_end = find(diff(ind_class)>1);
    trans_start = trans_end+1;
    trans_start = [ind_class(1), ind_class(trans_start)'];
    trans_end = [ind_class(trans_end)', ind_class(end)];

%     for n = 1:Nsamp
%         for j = 1:length(trans_start),
%             ind_sample = [ind_sample randi([trans_start(j), trans_end(j)],1)];
%         end
%         
%     end
    %taking all samples from one trial (specified by trial)
    ind_sample = [ind_sample, trans_start(trial):trans_end(trial)];
    
end


Xl = X(ind_sample,:);
Yl = Y(ind_sample);

inds = 1:size(X,1);
ind_nosample = inds(~ismember(inds, ind_sample));

Xu = X(ind_nosample, :);
Yu = Y(ind_nosample);

% repeating training data to balance the total number of labeled vs
% unlabeled
% Xl = repmat(Xl, ceil(eps+RepFac*length(Yu)/length(Yl)), 1);
% Yl = repmat(Yl, ceil(eps+RepFac*length(Yu)/length(Yl)), 1);

fprintf('%d labeled samples\n', length(ind_sample));   
fprintf('%d unlabeled samples\n', length(ind_nosample));
% fprintf('%d labeled samples after balancing\n', length(Yl));


%% Params to init forests

PARAM = {ntrees,T0,alpha,tau,Xl,Yl,Xu,Yu,n_class};

forest = ssforest(PARAM);

end