comp_type_ordering = ["Interaction", "First", "Last",
                      "Equals", "And", "Xor", "Choose", "Longer Choose",
                      "Shorter Choose", "After", "Before", "While",
                      "Between", "Overall"]

subq_type_ordering = ["Object Exists", "Relation Exists", "Interaction",
                      "Interaction Temporal Localization", "Exists Temporal Localization",
                      "First/Last", "Longest/Shortest Action",
                      "Conjunction", "Choose", "Equals", "Overall"]

banned_qtypes = {"Object", "Action After", "Action Before", "Action While",
                 "Action Between", "First Action", "Last Action",
                 "Object After", "Object Before", "Object While", "Object Between", "Action"}

collapsed_qtypes = {"Object Exists" : "Object Exists",
                    "Relation Exists" : "Relation Exists",
                    "Interaction" : "Interaction",
                    "Object" : "Object",
                    "Action" : "Action",
                    "Interaction After" : "Interaction Temporal Localization",
                    "Interaction Before" : "Interaction Temporal Localization",
                    "Interaction While" : "Interaction Temporal Localization",
                    "Interaction Between" : "Interaction Temporal Localization",
                    "Exists After" : "Exists Temporal Localization",
                    "Exists Before" : "Exists Temporal Localization",
                    "Exists While" : "Exists Temporal Localization",
                    "Exists Between" : "Exists Temporal Localization",
                    "Object After" : "Object Temporal Localization",
                    "Object Before" : "Object Temporal Localization",
                    "Object While" : "Object Temporal Localization",
                    "Object Between" : "Object Temporal Localization",
                    "Action After" : "Action Temporal Localization",
                    "Action Before" : "Action Temporal Localization",
                    "Action While" : "Action Temporal Localization",
                    "Action Between" : "Action Temporal Localization",
                    "And" : "Conjunction",
                    "Xor" : "Conjunction",
                    "Choose" : "Choose",
                    "Object Equals" : "Equals",
                    "Action Equals" : "Equals",
                    "First Object" : "First/Last",
                    "Last Object" : "First/Last",
                    "First Action" : "First/Last",
                    "Last Action" : "First/Last",
                    "Longest Action" : "Longest/Shortest Action",
                    "Shortest Action" : "Longest/Shortest Action",
                    "Longer Choose" : "Choose",
                    "Shorter Choose" : "Choose"
                    }

consistency_per_rule = {
    'Interaction Yes':{'Wrong':0, 'Total':0}, 'Interaction No':{'Wrong':0, 'Total':0},
    'Equals Yes':{'Wrong':0, 'Total':0}, 'Equals No': {'Wrong':0, 'Total':0},
    'And Yes':{'Wrong':0, 'Total':0}, 'And No': {'Wrong':0, 'Total':0},
    'Xor Yes':{'Wrong':0, 'Total':0}, 'Xor No': {'Wrong':0, 'Total':0},
    'Choose Temporal':{'Wrong':0, 'Total':0}, 'Choose Object': {'Wrong':0, 'Total':0},
    'After Yes':{'Wrong':0, 'Total':0}, 'After No': {'Wrong':0, 'Total':0},
    'Before Yes':{'Wrong':0, 'Total':0}, 'Before No': {'Wrong':0, 'Total':0},
    'While Yes':{'Wrong':0, 'Total':0}, 'While No': {'Wrong':0, 'Total':0},
    'Between Yes':{'Wrong':0, 'Total':0}, 'Between No': {'Wrong':0, 'Total':0}
}

parent_to_rules = {
    'Interaction' : {'Interaction'},
    'Exists Temporal Localization' : {'Between', 'While', 'After', 'Before'},
    'Interaction Temporal Localization' : {'While', 'Between', 'After', 'Interaction', 'Before'},
    'Conjunction' : {'Xor', 'And'},
    'Equals' : {'Equals'},
    'Choose' : {'Choose'},
    'Relation Exists' : {'Interaction'}
}

yesNo = [
    'Object Exists', 'Relation Exists', 'Interaction', 'Interaction After', 'Interaction Before', 'Interaction While', 'Interaction Between',
    'Exists After', 'Exists Before', 'Exists While', 'Exists Between', 'And', 'Xor'
]

