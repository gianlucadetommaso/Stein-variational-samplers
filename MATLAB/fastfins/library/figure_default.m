
% Set defaults for axes 
set(0,'defaultaxesfontname','times')
set(0,'defaultaxesfontsize',12)
set(0,'defaultaxeslinewidth',0.35)
set(0,'defaultaxesticklength',[0.005 0.025]);
set(0,'defaultaxesposition',[.08 .1 .88 .84]);
%set(0,'defaultaxeslinestyleorder',['- ';': ';'--']);
set(0,'defaultaxeslinestyleorder','-');

% Set defaults for figures
% This is for the LaTeX B5 style option in phrep.cls
%%% Change this for each figure (here \textwidth=12.5cm):
set(0,'defaultfigurepapertype','A4letter')
set(0,'defaultfigurepaperunits','cent')
set(0,'defaultfigurepaperposition',[3 10 9 6])
set(0,'defaultfigureposition',[1 2 700 550])
%set(0,'defaultfiguremenubar','on')
% set(0,'defaultfigureposition',[520 30 485 284])

% Set defaults for line-type objects
set(0,'defaultlinelinewidth',1.2)
set(0,'defaultlinemarkersize',5)
set(0,'DefaultTextInterpreter', 'latex')
return

%% True figure size in the below: width*height = 12.5*7.5
%% do this before "print -deps fig.eps"
set(gcf,'paperposition',[3 10 12.5 7.5]);


%% Good linewidths for multiple line plots:
LSO=['- ';'- ';'- '];
LW=[.90 .55 .15]';
for ii=1:3
  currpl=plot(EVecs(:,ii),LSO(ii));
  hold on
  set(currpl,'linewidth',LW(ii));
end



