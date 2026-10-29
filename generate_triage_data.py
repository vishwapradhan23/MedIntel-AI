"""
=============================================================================
MedIntel AI: Symptom Triage Data Generator
Script: generate_triage_data.py
Description: Generates synthetic training dataset (~200 samples per class across
             8 clinical specialties) and realistic, hand-crafted patient test
             dataset (>=20 unique samples per class) with varied templates,
             durations, informal phrasing, mixed symptoms, and negations.
=============================================================================
"""

import os
import random
import csv

# Set deterministic random seed
random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

SPECIALTIES = [
    "General Physician",
    "Cardiology",
    "Neurology",
    "Gastroenterology",
    "Dermatology",
    "Orthopedics",
    "Pulmonology",
    "Endocrinology"
]

# Vocabulary components for training data synthesis
DURATIONS = [
    "since this morning", "since yesterday", "since 2 days", "since 3 days",
    "for the past 4 days", "for about a week", "for the last two weeks",
    "for nearly a month", "on and off for several days", "started suddenly today",
    "since last night", "for over 5 days now", "intermittently for 2 weeks"
]

INTENSITIES = [
    "mild", "moderate", "severe", "slight", "intense", "sharp", "dull",
    "constant", "throbbing", "unbearable", "nagging", "persistent", "acute"
]

PREFIXES = [
    "I have been feeling", "I am suffering from", "Experiencing", "Patient reports",
    "Dealing with", "Having", "Complaining of", "Doctor, I have",
    "I've got", "Suffering from", "Suddenly developed", "Woke up with"
]

# Specialty symptom components
SPECIALTY_SYMPTOMS = {
    "General Physician": {
        "core": [
            "high fever", "mild low-grade fever", "body aches and chills", "extreme systemic fatigue",
            "loss of appetite", "general malaise", "shivering and sweating", "viral flu symptoms",
            "overall weakness", "lethargy and dizziness", "seasonal cold with body pain",
            "feeling run down and exhausted", "mild sore throat with feverish feeling"
        ],
        "contexts": [
            "and I feel completely drained", "with total lack of energy", "taking paracetamol with little relief",
            "unable to get out of bed", "cannot concentrate due to feeling so sick", "accompanied by generalized chills"
        ],
        "negations": [
            "no cough or chest pain but feeling high fever and body ache",
            "no stomach pain but continuous low fever and fatigue",
            "no vomiting but feeling chills, weakness and feverish",
            "not having breathing problems but complete exhaustion and shivering",
            "no rash anywhere but severe generalized body aches and fever"
        ]
    },
    "Cardiology": {
        "core": [
            "crushing chest pain", "tight pressure in chest", "chest discomfort radiating to left arm",
            "sharp pain in center of chest", "palpitations and racing heartbeat", "irregular pulse",
            "shortness of breath upon climbing stairs", "chest tightness while walking", "dizziness and fluttering heart",
            "swelling in both ankles and feet", "sensation of heart skipping beats", "sudden cold sweat with chest pressure"
        ],
        "contexts": [
            "especially when climbing up stairs", "worse upon exertion or fast walking", "radiating up towards my jaw and shoulder",
            "feeling heavy like an elephant sitting on my chest", "pulse feels fast even while resting in bed",
            "making me feel faint and dizzy"
        ],
        "negations": [
            "no cough or phlegm but sudden squeezing chest pain spreading to arm",
            "no stomach burn but severe chest tightness and rapid pounding heartbeat",
            "no fever or headache but palpitations and shortness of breath when walking",
            "not an acid reflux feeling but heavy pressure in chest and jaw discomfort",
            "no joint pain but racing pulse and severe tightness in my chest"
        ]
    },
    "Neurology": {
        "core": [
            "throbbing migraine headache on one side", "severe tension headache across temples",
            "sudden dizziness and spinning sensation", "vertigo when changing head position",
            "tingling and pins and needles in fingers", "numbness on left side of face",
            "uncontrollable tremors in hands", "loss of balance while walking straight",
            "sensitivity to bright light with visual aura", "sharp electric nerve pain in neck",
            "intermittent memory fog and confusion", "twitching in eyelid and facial muscles"
        ],
        "contexts": [
            "accompanied by light sensitivity and nausea", "room feels like it is spinning around me",
            "cannot maintain balance without holding walls", "making it difficult to hold a glass or write",
            "comes with flashing zigzag lights in vision", "sharp shooting sensations down the arms"
        ],
        "negations": [
            "no fever or chest pain but terrible pulsating headache on left side of head",
            "no muscle weakness but sudden spinning vertigo and nausea when lying down",
            "no skin rash but tingling sensation and numbness in right fingertips",
            "not feeling stomach pain but frequent tremors in both hands while resting",
            "no vision loss but severe throbbing headache with extreme sensitivity to light"
        ]
    },
    "Gastroenterology": {
        "core": [
            "severe abdominal cramps", "burning sensation behind breastbone acid reflux",
            "frequent watery diarrhea", "bloating and painful abdominal gas", "nausea and persistent vomiting",
            "sharp pain in lower right abdomen", "chronic constipation for several days",
            "stomach burning aggravated after meals", "loss of appetite with epigastric pain",
            "dark tarry stool and stomach irritation", "indigestion and excessive burping",
            "cramping pain around navel after eating spicy food"
        ],
        "contexts": [
            "especially worse after having meals", "antacids give only temporary mild relief",
            "accompanied by painful stomach gurgling", "cannot keep liquids down due to vomiting",
            "sharp stabbing sensation when pressing belly", "feeling bloated immediately after drinking water"
        ],
        "negations": [
            "no fever or cough but severe burning acid reflux and sour taste in mouth",
            "no chest pain but intense abdominal cramps and loose watery stools",
            "no headache but acute pain in lower right abdomen with nausea",
            "not a chest problem but constant heartburn and regurgitation of food",
            "no joint ache but bloating and sharp stomach cramps after eating"
        ]
    },
    "Dermatology": {
        "core": [
            "itchy red rash across neck and chest", "flaky dry scaly patches on elbows and knees",
            "raised red hives all over legs", "severe eczema flare-up on hands",
            "blisters with burning itchy sensation", "pigmented mole that has changed shape and color",
            "painful cystic acne breakouts across face", "peeling red skin that burns when sweating",
            "intense scalp itching with white flakes", "red circular ringworm rash on thigh",
            "chronic skin redness with visible tiny blood vessels", "cracked bleeding skin on knuckles"
        ],
        "contexts": [
            "itching becomes unbearable during night", "moisturizers are not helping soothe the irritation",
            "rash spreads when scratched", "skin feels hot, tender and irritated to touch",
            "leaving dry flaky residue on clothes", "started as small bumps and joined into red patches"
        ],
        "negations": [
            "no fever or joint pain but intense itchy red rash all over arms",
            "no stomach ache but dry scaly peeling skin patches on both knees",
            "no breathing trouble but sudden hives and welt outbreaks on back",
            "not an internal issue but a growing irregular dark mole on my shoulder",
            "no headache but severe burning itchy blistered skin on neck"
        ]
    },
    "Orthopedics": {
        "core": [
            "excruciating lower back pain", "severe right knee pain when walking down stairs",
            "stiff shoulder with restricted movement", "swollen twisted ankle from twisting foot",
            "cracking and grinding sensation in knee joint", "sharp pain radiating down buttocks to leg",
            "morning stiffness in fingers and knuckles", "difficulty standing up straight due to spine pain",
            "hip joint pain while walking or bending", "tender swelling around elbow tendon",
            "sharp pain in heel when taking first morning steps", "aching pain in wrist after repetitive typing"
        ],
        "contexts": [
            "makes it hard to walk or stand for long", "hurts severely when bending down or lifting weights",
            "stiffness lasts for over an hour in the morning", "pain shoots down into my calf and foot",
            "swollen joint feels warm and stiff", "cannot bear full body weight on the injured leg"
        ],
        "negations": [
            "no fever or cough but terrible sharp pain in lower back after lifting a box",
            "no skin rash but severe swelling and stiffness in left knee joint",
            "no chest pressure but unable to raise right arm due to acute shoulder pain",
            "not a stomach issue but excruciating hip joint pain whenever walking",
            "no headache or nausea but sprained ankle is swollen and painful to walk on"
        ]
    },
    "Pulmonology": {
        "core": [
            "chronic dry cough that won't go away", "wheezing sound when exhaling air",
            "severe shortness of breath after minimal exertion", "coughing up thick yellow-green phlegm",
            "chest congestion and bronchial tightness", "asthma flare-up with gasping for breath",
            "intermittent coughing spells waking me up at night", "shallow rapid breathing with tight chest",
            "crackling sound in chest while breathing deeply", "inability to take a full deep breath",
            "persistent smoker cough with mucus production", "burning airway sensation while breathing in cold air"
        ],
        "contexts": [
            "especially bad during nighttime and cold mornings", "inhaler provides only partial brief relief",
            "makes a whistling sound every time I breathe out", "cannot walk even 50 meters without gasping",
            "producing sticky phlegm continuously", "chest feels completely clogged and suffocating"
        ],
        "negations": [
            "no fever or sore throat but continuous dry wheezing cough for weeks",
            "no heart problem but severe shortness of breath and tight congested lungs",
            "no stomach ache but coughing up thick mucus and wheezing at night",
            "not feeling chest pressure but whistling sounds in lungs and breathless",
            "no headache or rash but constant bronchial congestion and dry coughing fits"
        ]
    },
    "Endocrinology": {
        "core": [
            "excessive unquenchable thirst and dry mouth", "frequent urination especially waking up at night",
            "unexplained rapid weight loss despite normal diet", "sudden unexplained weight gain and puffiness",
            "extreme sensitivity to cold temperatures", "excessive sweating and heat intolerance",
            "visible swelling in lower front of neck", "severe fatigue and sluggishness in morning",
            "brittle hair and dry cracking skin with low energy", "shaky trembling hands with rapid pulse and hunger",
            "constant cravings for sweets with high thirst", "irregular menstrual cycles with facial hair growth"
        ],
        "contexts": [
            "drinking several liters of water daily but still thirsty", "waking up four to five times every night to urinate",
            "lost 6 kg in three weeks without any dieting", "hands shake if I delay eating by even an hour",
            "feeling constantly exhausted despite sleeping 9 hours", "feeling freezing cold when others are comfortable"
        ],
        "negations": [
            "no fever or cough but drinking water constantly and peeing every hour",
            "no stomach pain but rapid unexplained weight loss of 5 kilos in two weeks",
            "no heart disease but extreme heat intolerance, sweating and neck swelling",
            "not a viral infection but severe lethargy, dry skin and feeling freezing cold",
            "no chest pain but shaky hands, excessive thirst and sudden blurred vision"
        ]
    }
}

# Hand-crafted test sentences (strictly separate from templates, informal, typos, varying lengths)
TEST_SET = [
    # General Physician (21 samples)
    ("I have been running a mild fever since Tuesday and my whole body aches, feeling super tired", "General Physician"),
    ("High fever 102F since yesterday with severe shivering, cold sweats and no energy to move", "General Physician"),
    ("feeling general weakness and low grade temperature since 3 days, throat slightly irritated", "General Physician"),
    ("woke up feeling completely exhausted, slight feverish feeling and sore muscles everywhere", "General Physician"),
    ("Just feeling sick all over with chills and body fatigue, no specific cough or stomach issue", "General Physician"),
    ("I think I caught viral flu, constant shivering, mild fever and heavy head since 2 days", "General Physician"),
    ("Temperature fluctuates between 99 and 101, body feels like heavy stone, loss of appetite", "General Physician"),
    ("fevver since last night with chills and sweating profusely, headache and joint pain all over", "General Physician"),
    ("Feeling totally drained, mild feverish sensation in evenings and extreme lethargy", "General Physician"),
    ("My entire body is sore and I have intermittent low grade fever for four days now", "General Physician"),
    ("sudden onset of high temp, shivering teeth, generalized muscle ache and dehydration", "General Physician"),
    ("no vomiting or cough but feeling weak, nauseous from fatigue and fever of 100.5", "General Physician"),
    ("General malaise, feeling sick since weekend with sweating and mild body chills", "General Physician"),
    ("Persistent fatigue and feeling warm to touch, muscles ache when walking around", "General Physician"),
    ("Doctor, I feel feverish every evening with heavy body pain and poor appetite", "General Physician"),
    ("Feeling completely beaten down by viral illness, low fever and constant shivering", "General Physician"),
    ("Mild chills, slight fever around 100, whole body feels achey and sluggish", "General Physician"),
    ("Woke up drenched in sweat with chills and muscular weakness throughout my torso and limbs", "General Physician"),
    ("no chest pain or rash, just classic flu symptoms like body stiffness, fatigue and fever", "General Physician"),
    ("running temperature since yesterday morning, taking paracetamol but fever bounces back", "General Physician"),
    ("Extremely tired, mild shivers, general weakness and loss of taste since 3 days", "General Physician"),

    # Cardiology (21 samples)
    ("Chest feels tight and uncomfortable, pain spreading to my left shoulder and arm", "Cardiology"),
    ("Experiencing sudden chest tightness with mild breathlessness while walking up the stairs", "Cardiology"),
    ("Heart is racing very fast even when sitting still on the sofa, feeling palpitation", "Cardiology"),
    ("heavy crushing feeling in the center of my chest since 30 minutes, sweating cold", "Cardiology"),
    ("Notice my pulse fluttering irregularly and skipping beats, makes me lightheaded", "Cardiology"),
    ("Sharp twinges across chest area when walking briskly, radiating slightly into jaw", "Cardiology"),
    ("Squeezing pressure in chest, shortness of breath, feeling like an elephant on ribs", "Cardiology"),
    ("Heart rate suddenly spikes to 130 bpm without doing any exercise, feels scary", "Cardiology"),
    ("Pounding heartbeat in my ears and chest discomfort that worsens when climbing stairs", "Cardiology"),
    ("Swelling around my ankles and feet, feeling breathless when lying flat on back", "Cardiology"),
    ("Dull aching pressure in left chest wall accompanied by sweating and slight dizziness", "Cardiology"),
    ("no cough or flu but feeling intense pressure right behind breastbone spreading to jaw", "Cardiology"),
    ("irregular heartbeat sensations intermittently throughout the afternoon, feeling faint", "Cardiology"),
    ("Chest tightness with cold clammy hands and anxiety, pulse is bounding fast", "Cardiology"),
    ("pain radiating from middle chest into left arm and back since an hour ago", "Cardiology"),
    ("Short of breath on mild walking and tight constricting chest discomfort", "Cardiology"),
    ("fluttering sensations in chest that come out of nowhere, accompanied by dizziness", "Cardiology"),
    ("Crushing center chest ache, trouble catching breath, dizzy when standing up", "Cardiology"),
    ("heart beats very rapidly after small meals, throbbing sensation in chest", "Cardiology"),
    ("both feet swollen and breathless while sleeping flat at night, heart feels overworked", "Cardiology"),
    ("Severe pressure across chest, radiating into left neck and arm, cold sweats", "Cardiology"),

    # Neurology (21 samples)
    ("Splitting headache on the right side of my head with nausea and light sensitivity", "Neurology"),
    ("Dizziness and room spinning sensation whenever I tilt my head or stand up fast", "Neurology"),
    ("Tingling and numbness in the fingers of my left hand since past two days", "Neurology"),
    ("Severe throbbing migraine behind my left eye, sound and bright lights make it unbearable", "Neurology"),
    ("noticed involuntary shaking in my right hand fingers while resting my arm on the table", "Neurology"),
    ("Sudden vertigo episode, feeling like the floor is dropping from underneath my feet", "Neurology"),
    ("Shooting electric nerve pain from my neck radiating down into my shoulder blade", "Neurology"),
    ("Difficulty balancing while walking in a straight line, feeling clumsy and uncoordinated", "Neurology"),
    ("One side of my forehead and face feels completely numb and prickly like pins and needles", "Neurology"),
    ("Intense headache on forehead temples, visual blind spots and zig zag lines in vision", "Neurology"),
    ("no fever but sudden episode of spinning head and nausea when getting out of bed", "Neurology"),
    ("My hands shake uncontrollably when trying to write or hold a cup of tea", "Neurology"),
    ("Wicked migraine for 24 hours, sensitivity to sound, nausea and throbbing temples", "Neurology"),
    ("Loss of sensation in my left foot toes and tingling numbness spreading up ankle", "Neurology"),
    ("Brain fog, mild confusion, and sharp localized head pain on the crown of my skull", "Neurology"),
    ("Constant dizziness, unsteady on my feet, feeling off balance when walking in hallway", "Neurology"),
    ("Trembling in fingers and stiffness in neck with severe tension headache across forehead", "Neurology"),
    ("Shooting nerve sensations down both arms when turning my neck, feeling numbness", "Neurology"),
    ("Sudden onset of severe vertigo, spinning surroundings, vomiting from motion sickness", "Neurology"),
    ("Throbbing unilateral headache, seeing flashing spots and cannot tolerate daylight", "Neurology"),
    ("Prickling pins and needles feeling all over my left cheek and jaw since morning", "Neurology"),

    # Gastroenterology (21 samples)
    ("Burning sensation in my chest and throat after eating meals, severe acid reflux", "Gastroenterology"),
    ("Intense stomach cramps and watery loose motions since midnight, feeling dehydrated", "Gastroenterology"),
    ("Sharp stabbing pain in lower right side of my belly, hurts when pressing down", "Gastroenterology"),
    ("Constant abdominal bloating, gas, and stomach ache every time I eat dairy or gluten", "Gastroenterology"),
    ("Nausea, vomiting everything I drink, and crampy pain in the upper stomach area", "Gastroenterology"),
    ("Severe heartburn and sour tasting liquid coming up into my mouth when lying down", "Gastroenterology"),
    ("Have not passed stool for five days, severe constipation and uncomfortable bloated belly", "Gastroenterology"),
    ("Persistent dull ache in upper abdomen that radiates through to my back after fatty food", "Gastroenterology"),
    ("Frequent loose stools 6 times today with painful abdominal cramping and rumbling", "Gastroenterology"),
    ("Stomach burning like fire after coffee or spicy food, antacid tabs barely work", "Gastroenterology"),
    ("no chest problem but bad acid reflux burning up my esophagus and throat", "Gastroenterology"),
    ("Sharp belly ache around navel moving downwards, nausea and zero appetite", "Gastroenterology"),
    ("Vomiting bile and having severe stomach pain after eating dinner yesterday", "Gastroenterology"),
    ("Excessive gas, painful swelling of belly, indigestion and irregular bowel movements", "Gastroenterology"),
    ("Acid regurgitation keeps waking me up coughing at night, burning upper abdomen", "Gastroenterology"),
    ("Continuous cramps in lower abdomen with frequent urgent diarrhea since yesterday", "Gastroenterology"),
    ("Pain in epigastric area, feels like gnawing hunger pain even right after lunch", "Gastroenterology"),
    ("Severe bloating and feeling full after eating just two bites of food, nausea", "Gastroenterology"),
    ("Sharp colicky pain in right lower abdomen accompanied by slight nausea and chills", "Gastroenterology"),
    ("Chronic indigestion, heartburn behind breastbone and foul smelling burps", "Gastroenterology"),
    ("Stomach cramps and persistent vomiting since morning, unable to keep water down", "Gastroenterology"),

    # Dermatology (21 samples)
    ("Extremely itchy red rash with small bumps all over my forearms and neck", "Dermatology"),
    ("Dry flaky red patches on my elbows that peel and bleed when scratched", "Dermatology"),
    ("Woke up with raised red hives covering both my thighs and back, burning and itching", "Dermatology"),
    ("Severe flare-up of eczema on both hands, skin is cracked, red and very tender", "Dermatology"),
    ("Notice a dark mole on my shoulder that has irregular jagged borders and got bigger", "Dermatology"),
    ("Painful cystic acne breakouts along my jawline and cheeks that won't clear up", "Dermatology"),
    ("Scalp is intensely itchy with thick white dandruff flakes falling onto my shoulders", "Dermatology"),
    ("Circular red rash with clear center on my thigh, looks like ringworm and itches", "Dermatology"),
    ("Skin on my face is constantly flushed red with small visible blood vessels on nose", "Dermatology"),
    ("Blistering itchy rash on the palms of my hands, skin is peeling off in layers", "Dermatology"),
    ("no internal illness but skin on my neck is covered in red itchy welts and hives", "Dermatology"),
    ("Rough scaly plaques on both knees that itch like crazy during nighttime", "Dermatology"),
    ("Deep painful pimples under skin across forehead and back, leaving dark marks", "Dermatology"),
    ("Dry cracked skin between toes that burns and itches terribly after wearing shoes", "Dermatology"),
    ("Sudden breakout of red itchy bumps across torso after using a new scented soap", "Dermatology"),
    ("An existing mole on my back turned darker black and started itching occasionally", "Dermatology"),
    ("Red burning rash in skin folds under arms and groin, aggravated by sweating", "Dermatology"),
    ("Skin peeling off my fingertips and knuckles, very dry, irritated and painful fissures", "Dermatology"),
    ("Cluster of tiny water blisters on lips and chin that sting and crust over", "Dermatology"),
    ("Chronic red itchy patches on cheeks that flare up whenever exposed to sun", "Dermatology"),
    ("Spreading red rash across chest with intense itchiness, antihistamines not working", "Dermatology"),

    # Orthopedics (21 samples)
    ("Severe lower back pain after lifting heavy luggage, can hardly bend or walk", "Orthopedics"),
    ("Sharp pain in my right knee joint whenever I try to walk downstairs or kneel", "Orthopedics"),
    ("Twisted my ankle playing football yesterday, it is swollen, bruised and hurts to step on", "Orthopedics"),
    ("Shoulder joint is so stiff that I cannot raise my arm above head to comb my hair", "Orthopedics"),
    ("Sharp shooting pain starting in lower back and going all the way down my right leg", "Orthopedics"),
    ("Cracking sound and grinding pain in both knees whenever I stand up from a chair", "Orthopedics"),
    ("Stiff painful fingers and wrist joints every morning, takes an hour to loosen up", "Orthopedics"),
    ("Severe pain in left hip joint when walking more than 100 meters, limping slightly", "Orthopedics"),
    ("Sharp stabbing heel pain during my very first steps out of bed in the morning", "Orthopedics"),
    ("Outer elbow joint is tender and throbs with pain when gripping objects or typing", "Orthopedics"),
    ("no fever but excruciating pain in lumbar spine after bending down to pick up groceries", "Orthopedics"),
    ("Swollen warm right knee with fluid, cannot bend it past 90 degrees without agony", "Orthopedics"),
    ("Severe neck and upper back stiffness, cannot turn head to check blind spot while driving", "Orthopedics"),
    ("Sprained wrist after falling on an outstretched hand, tender swelling and weakness", "Orthopedics"),
    ("Sciatica nerve pain shooting through left buttock down into calf and foot", "Orthopedics"),
    ("Knee gives way occasionally while walking, followed by sharp clicking pain inside joint", "Orthopedics"),
    ("Inability to bear weight on right foot after awkward twist, ankle is purple and swollen", "Orthopedics"),
    ("Chronic dull ache in lower spine that radiates into hips when sitting at desk", "Orthopedics"),
    ("Shoulder impingement pain when reaching behind back or sleeping on my side", "Orthopedics"),
    ("Both knees feel completely worn out, bone on bone sensation and stiffness", "Orthopedics"),
    ("Acute spasm in lower back muscles, trapped on floor unable to stand upright", "Orthopedics"),

    # Pulmonology (21 samples)
    ("Persistent dry cough that has been hacking away for over three weeks now", "Pulmonology"),
    ("Wheezing and whistling sounds coming from my chest whenever I breathe out", "Pulmonology"),
    ("Shortness of breath after walking just a short distance, feel like I can't get enough air", "Pulmonology"),
    ("Coughing up thick yellowish green phlegm every morning with heavy chest congestion", "Pulmonology"),
    ("Asthma is acting up badly, tight constricted lungs and needing rescue inhaler repeatedly", "Pulmonology"),
    ("Deep chest rattle when breathing in, feel like my bronchial tubes are filled with mucus", "Pulmonology"),
    ("Cannot take a satisfactory deep breath, chest feels tight and suffocated since yesterday", "Pulmonology"),
    ("Coughing fits so severe they make me gag and leave my rib cage aching", "Pulmonology"),
    ("Breathless when walking up a gentle slope, wheezing sound in lungs on exhalation", "Pulmonology"),
    ("Heavy smoker with a chronic morning cough and bringing up greyish phlegm daily", "Pulmonology"),
    ("no fever or throat pain but continuous wheezing cough and tight airways for a month", "Pulmonology"),
    ("Chest feels clogged with mucus, whistling noises while breathing at night in bed", "Pulmonology"),
    ("Short of breath doing routine housework, gasping for air after climbing four steps", "Pulmonology"),
    ("Dry spasmodic cough that gets triggered by cold weather and dusty air", "Pulmonology"),
    ("Bronchial tightness and continuous tickling cough deep inside my chest cavity", "Pulmonology"),
    ("Waking up in the middle of the night gasping for breath with tight wheezing lungs", "Pulmonology"),
    ("Coughing non-stop for 10 days, chest feels bruised and breathing feels labored", "Pulmonology"),
    ("Lungs feel tight like someone strapped a belt around my rib cage, wheezing sound", "Pulmonology"),
    ("Productive cough bringing up copious mucus, lungs feel heavy and congested", "Pulmonology"),
    ("Shortness of breath that came on over the week, whistling lungs and dry persistent cough", "Pulmonology"),
    ("Severe asthmatic wheeze, struggling to exhale fully, chest feels restricted and tight", "Pulmonology"),

    # Endocrinology (21 samples)
    ("Drinking gallons of water every day but mouth is still dry, peeing constantly at night", "Endocrinology"),
    ("Unexplained weight loss of 6 kilograms in three weeks even though I eat plenty of food", "Endocrinology"),
    ("Always feeling freezing cold even when the heater is on, extreme sluggishness and fatigue", "Endocrinology"),
    ("Sweating profusely, heat intolerance, and hands trembling when holding papers", "Endocrinology"),
    ("Notice a visible lump and swelling on the front of my neck, throat feels full", "Endocrinology"),
    ("Uncontrolled weight gain despite eating very little, hair is brittle and skin is dry", "Endocrinology"),
    ("Blood sugar reading was 240 this morning, feeling thirsty, tired and blurry vision", "Endocrinology"),
    ("Waking up 5 times every night to urinate and feeling insatiable thirst all day long", "Endocrinology"),
    ("Rapid heart flutter, jittery hands, nervousness and losing weight without trying", "Endocrinology"),
    ("Severe fatigue, brain fog, cold sensitivity, and neck feels slightly enlarged", "Endocrinology"),
    ("no cough or cold but constant unquenchable thirst and going to bathroom every 40 minutes", "Endocrinology"),
    ("Lost 15 pounds in a month without diet or workout, hands shake and pulse is fast", "Endocrinology"),
    ("Severe heat intolerance, sweating through my clothes indoors, and anxious shaking", "Endocrinology"),
    ("Extreme lethargy in the morning, puffy swollen face, and sensitivity to cold weather", "Endocrinology"),
    ("Constant hunger and thirst combined with rapid weight loss and sudden vision blurriness", "Endocrinology"),
    ("Thyroid gland on neck looks swollen, difficulty swallowing solid foods comfortably", "Endocrinology"),
    ("Hair thinning rapidly, skin extremely dry and flaky, feeling exhausted after 8 hrs sleep", "Endocrinology"),
    ("Excessive thirst, drinking water by the bottle, dry tongue and frequent nocturnal urination", "Endocrinology"),
    ("Jittery trembling fingers, racing resting pulse, heat sensitivity and unexpected weight drop", "Endocrinology"),
    ("High blood glucose readings on home monitor, feeling lightheaded with intense dry mouth", "Endocrinology"),
    ("Puffiness around eyes, cold intolerance, slow heart rate and chronic heavy tiredness", "Endocrinology")
]

def generate_training_data(target_per_specialty=200):
    """
    Synthesize varied training samples per specialty using templates,
    symptom vocabularies, durations, intensities, and negations.
    """
    train_records = []

    for specialty in SPECIALTIES:
        spec_data = SPECIALTY_SYMPTOMS[specialty]
        cores = spec_data["core"]
        contexts = spec_data["contexts"]
        negations = spec_data["negations"]

        records_for_class = []

        # 1. Include negated sentences with variations
        for neg in negations:
            records_for_class.append(neg)
            # Add with duration
            dur = random.choice(DURATIONS)
            records_for_class.append(f"{neg} {dur}")
            # Add with prefix
            pref = random.choice(PREFIXES)
            records_for_class.append(f"{pref} {neg.lower()}")

        # 2. Template syntheses until target_per_specialty is reached
        while len(records_for_class) < target_per_specialty:
            core = random.choice(cores)
            ctx = random.choice(contexts)
            dur = random.choice(DURATIONS)
            intensity = random.choice(INTENSITIES)
            prefix = random.choice(PREFIXES)

            pattern = random.randint(1, 8)
            if pattern == 1:
                # Prefix + core + duration
                text = f"{prefix} {core} {dur}."
            elif pattern == 2:
                # Intensity + core + context
                text = f"{intensity.capitalize()} {core} {ctx} {dur}."
            elif pattern == 3:
                # Core + context
                text = f"Dealing with {core}, {ctx}."
            elif pattern == 4:
                # Multiple core symptoms
                second_core = random.choice([c for c in cores if c != core])
                text = f"{prefix} {core} along with {second_core} {dur}."
            elif pattern == 5:
                # Patient statement
                text = f"Doctor, I've had {intensity} {core} {dur} {ctx}."
            elif pattern == 6:
                # Short informal statement
                text = f"{core} {dur}, {ctx}."
            elif pattern == 7:
                # Question style
                text = f"Having {intensity} {core} {dur}, {ctx}."
            else:
                # Mixed phrasing
                second_core = random.choice([c for c in cores if c != core])
                text = f"{core} and {second_core} {dur}."

            # Normalize string
            clean_text = " ".join(text.split()).strip()
            if clean_text not in records_for_class:
                records_for_class.append(clean_text)

        # Truncate to exact target
        records_for_class = records_for_class[:target_per_specialty]
        for text in records_for_class:
            train_records.append({"text": text, "specialty": specialty})

    # Shuffle training data
    random.shuffle(train_records)
    return train_records

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    train_path = os.path.join(DATA_DIR, "triage_train.csv")
    test_path = os.path.join(DATA_DIR, "triage_test.csv")

    print("=" * 60)
    print("   GENERATING SYMPTOM TRIAGE DATASETS (TRAIN & TEST)      ")
    print("=" * 60)

    # 1. Generate Training Data (~200 per specialty = ~1600 rows)
    train_records = generate_training_data(target_per_specialty=200)
    with open(train_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "specialty"])
        writer.writeheader()
        writer.writerows(train_records)
    print(f"Generated training dataset: {len(train_records)} rows -> {train_path}")

    # Verify class balance
    class_counts = {}
    for r in train_records:
        spec = r["specialty"]
        class_counts[spec] = class_counts.get(spec, 0) + 1
    for spec, count in sorted(class_counts.items()):
        print(f"  - {spec}: {count} records")

    # 2. Write Test Data (>=20 hand-crafted patient sentences per specialty = >=160 rows)
    test_records = [{"text": text, "specialty": spec} for text, spec in TEST_SET]
    # Shuffle test records
    random.shuffle(test_records)
    with open(test_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "specialty"])
        writer.writeheader()
        writer.writerows(test_records)
    print(f"\nWritten test dataset: {len(test_records)} rows -> {test_path}")

    test_counts = {}
    for r in test_records:
        spec = r["specialty"]
        test_counts[spec] = test_counts.get(spec, 0) + 1
    for spec, count in sorted(test_counts.items()):
        print(f"  - {spec}: {count} records")

    print("\nDatasets generated and verified successfully!")

if __name__ == "__main__":
    main()
