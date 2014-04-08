close all
clear all
epochs = 4;
tau = [50 60 70 80 90 100];
% alpha = [0.75 1.25 1.5]; 
subj = 21;   %total # of train subjects
% F = cell(length(tau),1);  %cell array containing all the trained forests
acc = zeros(length(tau),epochs+1,subj); %matrix with accuracy for every subject

tic

%main loop over # of train vs test subjects
for Ntr = 1:subj
    
    sprintf('Train subjects %d', num2str(Ntr));

    %loop over model parameters (tau, alpha, etc)
    for k = 1:length(tau)
        
        F = configUCI(Ntr,9);     %initialize a forest - specify Ntr and Nte subjects
        
        F.tau = tau(k);          %set tau
        %     F{k}.alpha = alpha(k);    %set alpha
        
        F.trainforest_multic(epochs);    %train ssrf
        
        acc(k,:,Ntr) = F.acc;   %save accuracy over parameters
        
        clear F
        
    end
    
end

toc

save('accforests.mat',acc);

% save('trainedforests_40trees_1tr_6te_tau70.mat','F')

% figure, hold on
% plot(acc','LineWidth',2)
% for s = 1:length(alpha)
%     l{s} = num2str(alpha(s));
% end
% 
% legend(l)