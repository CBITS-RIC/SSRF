function F = multitrain

close all
clear all
epochs = 200;
acc = [];
% tau = [70 80];
alpha = [0.75 1.25 1.5]; 
N = length(alpha);    %# of simulations to run

F = cell(N,1);  %cell array containing all the trained forests
tic
for k = 1:N
   
    F{k} = config;
    
%     F{k}.tau = tau(k);
    F{k}.alpha = alpha(k);
    
    F{k}.trainforest_multic(epochs);
    
    acc = [acc;F{k}.acc];
%     tau = [tau;F{k}.tau];
    
end
t= toc

% save('trainedforests_40trees_1tr_6te_tau70.mat','F')

figure, hold on
plot(acc','LineWidth',2)
for s = 1:length(alpha)
    l{s} = num2str(alpha(s));
end

legend(l)