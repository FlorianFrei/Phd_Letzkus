function Florian_2

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


MaxTrials = 48;


% Define the maximum number of trials
% Define trial types (you can modify this as needed)
sequence = [ones(1, 12), 2*ones(1, 12), 3*ones(1,12),4*ones(1, 12)];

% Shuffle the array randomly
sequence = sequence(randperm(MaxTrials));


TrialTypes = sequence;
%TrialTypes = randsrc(1, MaxTrials, [1 3; 0.5 0.5]);

%% Plot
BpodSystem.ProtocolFigures.OutcomePlotFig = figure('Position', [50 540 1000 250],'name','Outcome plot','numbertitle','off', 'MenuBar', 'none', 'Resize', 'off');
BpodSystem.GUIHandles.OutcomePlot = axes('Position', [.075 .3 .89 .6]);
TrialTypeOutcomePlot(BpodSystem.GUIHandles.OutcomePlot,'init',TrialTypes);
% Initialize parameter GUI plugin

% Create the outcome plot



% Loop through trials
for currentTrial = 1:MaxTrials

    switch TrialTypes(currentTrial) % Determine trial-specific state matrix fields
        case 1
            cue = 'Upsweep';

        case 2
            cue = 'Downsweep';

        case 3
            cue = 'Opto_Upwsweep_LS';

        case 4
            cue = 'Opto_Downsweep_LS';

    end

    % Create a new state machine (SMA) for the current trial
    sma = NewStateMatrix();  % Create a blank matrix to define the trial's finite state machine

    % ITI state: Inter-Trial Interval (randomized duration)

    sma = AddState(sma, 'Name', 'ITI', ...
        'Timer',  7,...  % Random duration (mean 1s, SD 0.5s)
        'StateChangeConditions', {'Tup', cue},...
         'OutputActions', {'HiFi1','*'});  % Any output action needed in ITI

    sma = AddState(sma, 'Name', 'Upsweep', ...
        'Timer', 5,...  % Duration of cue
        'StateChangeConditions', {'Tup', 'exit'},...
         'OutputActions', {'HiFi1',1});


    sma = AddState(sma, 'Name', 'Downsweep', ...
        'Timer', 5,...  % Duration of cue
        'StateChangeConditions', {'Tup', 'exit'},...
        'OutputActions', {'HiFi1',2});



    sma = AddState(sma, 'Name', 'Opto_Upwsweep_LS', ...
        'Timer',0.1,...  % Duration of cue
        'StateChangeConditions', {'Tup', 'Opto_Upwsweep'},...
        'OutputActions', {'WavePlayer1',1});

    sma = AddState(sma, 'Name', 'Opto_Upwsweep', ...
        'Timer', 5,...  % Duration of cue
        'StateChangeConditions', {'Tup', 'Laser_offramp'},...
         'OutputActions', {'HiFi1',1});



    sma = AddState(sma, 'Name', 'Opto_Downsweep_LS', ...
        'Timer', 0.1,...  % Duration of cue
        'StateChangeConditions', {'Tup', 'Opto_Downsweep'},...
        'OutputActions', {'WavePlayer1',1});

    sma = AddState(sma, 'Name', 'Opto_Downsweep', ...
        'Timer', 5,...  % Duration of cue
        'StateChangeConditions', {'Tup', 'Laser_offramp'},...
        'OutputActions', {'HiFi1', 2});


    sma = AddState(sma, 'Name', 'Laser_offramp', ...
        'Timer', 1,...  % Duration of cue
        'StateChangeConditions', {'Tup', 'exit'},...
        'OutputActions', {});




    % Send the state machine to the Bpod system and run the trial
    SendStateMachine(sma);
    RawEvents = RunStateMachine;

    if ~isempty(fieldnames(RawEvents)) % If trial data was returned
        BpodSystem.Data = AddTrialEvents(BpodSystem.Data, RawEvents);
        BpodSystem.Data.TrialSettings(currentTrial) = S; % Adds the settings used for the current trial to the Data struct (to be saved after the trial ends)
        BpodSystem.Data.TrialTypes(currentTrial) = TrialTypes(currentTrial);
        SaveBpodSessionData;
        Outcomes = zeros(1,BpodSystem.Data.nTrials);

        for x = 1:BpodSystem.Data.nTrials

            if ~isnan(BpodSystem.Data.RawEvents.Trial{x}.States.ITI(1))

                Outcomes(x) = 1;

            else

                Outcomes(x) = 3;
            end
        end
        TrialTypeOutcomePlot(BpodSystem.GUIHandles.OutcomePlot,'update',BpodSystem.Data.nTrials+1,TrialTypes,Outcomes)
    end

    HandlePauseCondition; % Checks to see if the protocol is paused. If so, waits until user resumes.

    % Exit the session if the user has pressed the end button
    if BpodSystem.Status.BeingUsed == 0
        return
    end
end
