function multitrain

close all
clear all
N = 5;  %# of sims to run
epochs = 100;
F = cell(N,1);  %cell array containing all the trained forests
for k = 1:N
   
    F{k} = config;
    tau = F{k}.tau;
    F{k}.tau = tau*2*k;
    F{k}.trainforest(epochs);
    
end

save('trainedforests.mat','F')