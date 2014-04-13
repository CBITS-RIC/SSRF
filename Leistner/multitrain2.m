clc
close all
clear all
epochs = 100;
% tau = [10 20 30 40];
T0 = 1;
% alpha = [0.75 1.25 1.5]; 
subj = 3;   %total # of train (labeled) subjects to use
% F = cell(length(tau),1);  %cell array containing all the trained forests
acc = zeros(length(T0),epochs+1,subj); %matrix with accuracy for every subject
oobe = acc;                            %matrix with oobe for every subject and epoch
Nte = 3;    %# of subject to use as test (unlabeled)

%main loop over # of train vs test subjects
for Ntr = 3
    
    fprintf('# of Training subjects: %d \n', Ntr);

    %loop over model parameters (tau, alpha, etc)
    for k = 1%:length(tau)
        
        F = configUCI(Ntr,Nte);     %initialize a forest - specify Ntr and Nte subjects
        
        F.tau = 80;%tau(k);          %set tau (80)
        %     F{k}.alpha = alpha(k);    %set alpha
        F.T0 = T0;%(k);           %set T0
        
        tic 
        F.trainforest_multic(epochs);    %train ssrf

        toc
        
        acc(k,:,Ntr) = F.acc;   %save accuracy over parameters
        oobe(k,:,Ntr) = F.oobe; %save OOBE
        clear F
        
    end
    
end

toc

save 'accforests.mat' acc;
% save('trainedforests_40trees_1tr_6te_tau70.mat','F')

% figure, hold on
% plot(acc(:,:,3)','LineWidth',2)
% for s = 1:length(tau)
%     l{s} = num2str(tau(s));
% end
% 
% legend(l)