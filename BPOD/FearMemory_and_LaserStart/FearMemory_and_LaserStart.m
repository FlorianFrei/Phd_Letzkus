function FearMemory_and_LaserStart
[AllSounds{1},Fs] = audioread(['C:/Users/Admin/Desktop/Upsweep.wav']);
[AllSounds{2},Fs] = audioread(['C:/Users/Admin/Desktop/Downsweep.wav']);
global BpodSystem
W = BpodWavePlayer('COM7');
W.OutputRange = '0V:5V'; 
W.TriggerMode = 'Master';
W.SamplingRate =  29000;
myWave1 = load('C:/Users/Admin/Desktop/CustomWaveform.mat', 'waveform').waveform;
W.loadWaveform(1, myWave1)
%LoadSerialMessages('WavePlayer1', {['P' 1 0]});
H = BpodHiFi('COM10');
H.SamplingRate = Fs;
%H.DigitalAttenuation_dB = -5;
Upsweep = transpose(AllSounds{1});
Downsweep = transpose(AllSounds{2});
H.load(1, Upsweep);
H.load(3, Downsweep);
LoadSerialMessages('HiFi1', {['P' 0], ['P' 2]},'WavePlayer1', {['P' 1 0]});
LoadSerialMessages('HiFi1', {['P' 0], ['P' 2]});
LoadSerialMessages('WavePlayer1', {['P' 1 0]});
S = BpodSystem.ProtocolSettings; % settings that can be adjusted during the session

%% --- TRIAL CONFIGURATION MODIFICATIONS ---
MainTrials = 48;
BaselineTrials = 5;
MaxTrials = MainTrials + BaselineTrials; % Total is now 58

% Generate the main randomized sequence
sequence = [ones(1, 12), 2*ones(1, 12), 3*ones(1,12), 4*ones(1, 12)];
sequence = sequence(randperm(MainTrials));

% Prepend 10 baseline trials (Type 0) at the very beginning
TrialTypes = [zeros(1, BaselineTrials), sequence];
%% -----------------------------------------

%% Plot
BpodSystem.ProtocolFigures.OutcomePlotFig = figure('Position', [50 540 1000 250],'name','Outcome plot','numbertitle','off', 'MenuBar', 'none', 'Resize', 'off');
BpodSystem.GUIHandles.OutcomePlot = axes('Position', [.075 .3 .89 .6]);
TrialTypeOutcomePlot(BpodSystem.GUIHandles.OutcomePlot,'init',TrialTypes);

% Loop through trials
for currentTrial = 1:MaxTrials
    
    % Create a new state machine (SMA) for the current trial
    sma = NewStateMatrix();  
    
    switch TrialTypes(currentTrial)
        %% --- NEW BASELINE STATE ---
        case 0
            % Simple 10-second spacing state with no actions
            sma = AddState(sma, 'Name', 'Laser_Only', ...
                'Timer', 10,...  
                'StateChangeConditions', {'Tup', 'exit'},...
                'OutputActions', {'WavePlayer1',1});
                
        %% --- MAIN EXPERIMENT STATES ---
        case 1
            cue = 'Upsweep';
        case 2
            cue = 'Downsweep';
        case 3
            cue = 'Opto_Upsweep_LS';
        case 4
            cue = 'Opto_Downsweep_LS';
    end
    
    % Only build the main state flows if it's not a baseline trial
    if TrialTypes(currentTrial) > 0
        % ITI state: Inter-Trial Interval (randomized duration)
        sma = AddState(sma, 'Name', 'ITI', ...
            'Timer',  7,...  
            'StateChangeConditions', {'Tup', cue},...
             'OutputActions', {'HiFi1','*'});  

        sma = AddState(sma, 'Name', 'Upsweep', ...
            'Timer', 5,...  
            'StateChangeConditions', {'Tup', 'exit'},...
             'OutputActions', {'HiFi1',1});

        sma = AddState(sma, 'Name', 'Downsweep', ...
            'Timer', 5,...  
            'StateChangeConditions', {'Tup', 'exit'},...
            'OutputActions', {'HiFi1',2});

        sma = AddState(sma, 'Name', 'Opto_Upsweep_LS', ...
            'Timer',0.1,...  
            'StateChangeConditions', {'Tup', 'Opto_Upsweep'},...
            'OutputActions', {'WavePlayer1',1});

        sma = AddState(sma, 'Name', 'Opto_Upsweep', ...
            'Timer', 5,...  
            'StateChangeConditions', {'Tup', 'Laser_offramp'},...
             'OutputActions', {'HiFi1',1});

        sma = AddState(sma, 'Name', 'Opto_Downsweep_LS', ...
            'Timer', 0.1,...  
            'StateChangeConditions', {'Tup', 'Opto_Downsweep'},...
            'OutputActions', {'WavePlayer1',1});

        sma = AddState(sma, 'Name', 'Opto_Downsweep', ...
            'Timer', 5,...  
            'StateChangeConditions', {'Tup', 'Laser_offramp'},...
            'OutputActions', {'HiFi1', 2});
        
        sma = AddState(sma, 'Name', 'Laser_offramp', ...
            'Timer', 1,...  
            'StateChangeConditions', {'Tup', 'exit'},...
            'OutputActions', {});
    end
    
    % Send the state machine to the Bpod system and run the trial
    SendStateMachine(sma);
    RawEvents = RunStateMachine;
    
    if ~isempty(fieldnames(RawEvents)) % If trial data was returned
        BpodSystem.Data = AddTrialEvents(BpodSystem.Data, RawEvents);
        BpodSystem.Data.TrialSettings(currentTrial) = S; 
        BpodSystem.Data.TrialTypes(currentTrial) = TrialTypes(currentTrial);
        SaveBpodSessionData;
        
        Outcomes = zeros(1,BpodSystem.Data.nTrials);
        for x = 1:BpodSystem.Data.nTrials
            % Minor fix to handle outcomes checking for baseline trials safely
            if isfield(BpodSystem.Data.RawEvents.Trial{x}.States, 'ITI') && ...
               ~isnan(BpodSystem.Data.RawEvents.Trial{x}.States.ITI(1))
                Outcomes(x) = 1;
            elseif isfield(BpodSystem.Data.RawEvents.Trial{x}.States, 'BaselineWait')
                Outcomes(x) = 1; % Marks baseline trials as a placeholder "success/complete" color on plot
            else
                Outcomes(x) = 3;
            end
        end
        TrialTypeOutcomePlot(BpodSystem.GUIHandles.OutcomePlot,'update',BpodSystem.Data.nTrials+1,TrialTypes,Outcomes)
    end
    
    HandlePauseCondition; 
    if BpodSystem.Status.BeingUsed == 0
        return
    end
end