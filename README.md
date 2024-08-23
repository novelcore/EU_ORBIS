# EU_ORBIS

This repository contains the implementation of EU project: ORBIS

**Abstract:** ORBIS addresses the disconnects between ambitious ideas and collective actions at a large socio-technical scale. It responds to the profound lack of dialogue between citizenship and policy making institutions by providing a theoretically sound and highly pragmatic socio-technical solution to enable the transition to a more inclusive, transparent and trustful Deliberative Democracy in Europe. The project shapes and supports new democratic models that are developed through deliberative democracy processes; it follows a socioconstructive approach in which deliberative democracy is not a theory which prescribes new democratic practices and models, but rather the process through which we can collectively imagine and realize them. ORBIS provides new ways to understand and facilitate the emergence of new participatory democracy models, together with the mechanisms to scale them up and consolidate them at institutional level. It delivers: (i) a sound methodology for deliberative participation and co-creation at scale; (ii) novel AI-enhanced tools for deliberative participation across diverse settings; (iii) a novel socio-technical approach that augments the articulation between deliberative processes and representative institutions in liberal democracies; (iv) new evidence-based democratic models that emerge from the application of citizen deliberation processes; (v) demonstrated measurable impact of such innovations in real-world settings. The project builds on cutting-edge AI tools and technologies to develop a sustainable digital solution, and bridges theories and technological solutions from the fields of political and social science, social innovation, Artificial Intelligence, argumentation and digital democracy. The achievement of the project’s goal is validated through six use cases addressing contemporary issues at different scales and settings, experimenting with different civic participation and deliberation models, and involving diverse types of stakeholders.

## Table of Contents
- [EU\_ORBIS](#eu_orbis)
  - [Table of Contents](#table-of-contents)
  - [Summary](#summary)
  - [How to run](#how-to-run)
  - [Data](#data)
  - [Clustering](#clustering)
  - [Ontology and Knowledge Graph Construction](#ontology-and-knowledge-graph-construction)
  - [Contact](#contact)

## Summary

Keywords:
- Democratic engagement and civic participation
- New participatory democracy models

Project number: 101094765

Project name: Augmenting participation, co-creation, trust and transparency in Deliberative Democracy at all scales

Project acronym: ORBIS

Call: HORIZON-CL2-2022-DEMOCRACY-01

Topic: HORIZON-CL2-2022-DEMOCRACY-01-02

Type of action: HORIZON Research and Innovation Actions

Granting authority: European Research Executive Agency

Grant managed through EU Funding & Tenders Portal: Yes (eGrants)

Project starting date: fixed date: 1 February 2023

Project end date: 31 January 2026

Project duration: 36 months

## How to run
--- 
1. Clone a virtual environment 
```
    python3 -m  venv .env
```

2. Activate the virtual environment 
```
    source .env/bin/activate
```
3. Install requirements
```
    pip3 install -r requirements.txt
```

Download and install the `en_core_web_sm` model for spaCy:
```
    python3 -m spacy download en_core_web_sm
```

Install `LMRank` directly from source
```
pip3 install git+https://github.com/NC0DER/LMRank/
```

4. Run
```
    python3 0_download_bcause.py  # Downloads data from BCAUSE API
    python3 1_clustering.py       # Perfom clustering in data
    python3 2_text2KG.py          # Creates a KG from the user comments
```

Alternatively, you can run all the scripts sequentially with:
```
    sh bash.sh
```

**Note**:  Before executing the scripts, make sure to specify the following in the `config.yaml` file:

- **OpenAI API Key**
- **Neo4j Credentials**
- **Auth Token**
- **Embeddings Model (from Hugging Face)**

These configurations are essential for the clustering pipeline and for generating the Knowledge Graph.

## Data
This repository contains the implementation of ORBIS Project in **BCAUSE** dataset.
The dataset consists of user posts categorized into three categories: 
- **Positions:** The main stances or topics of discussion. 
- **Arguments in Favor:** Supporting arguments for each position. 
- **Arguments Against:** Counterarguments to each position.
<br><br><br>
## Clustering
Clustering is conducted on both In Favor and Against Statements for each Position, as illustrated in the diagram below:

<img src="images/ORBIS_Clustering.png" alt="BCAUSE Clustering"/>

In detail:
1. **Clustering**: KMeans algorithm is applied to cluster both "In Favor" and "Against" statements for each position.
2. **Summarization**: Each cluster is summarized using a combination of extractive and abstractive methods.
3. **Keyphrases**: The top 5 keyphrases are identified for each cluster using the [LMRank](https://ieeexplore.ieee.org/document/10179894) algorithm.
4. **Titles**: A concise title is generated for each cluster using the GPT model.
<br><br><br>

## Ontology and Knowledge Graph Construction

Post-clustering, the `text2KG` script constructs a Knowledge Graph based on the following ontology:

<img src="images/ORBIS_Ontology.png" height="900">

Here is the nodes with their propertirs in detail:

- **SUBJECT**
  - *id (STRING):* Unique identifier for the `SUBJECT` node.
  - *author_id (STRING):* Unique identifier for the author of the content.
  - *title (STRING):* The title of the subject.
  - *tagline (STRING):* A brief tagline or description for the subject.

- **POSITION**
  - *id (STRING):* Unique identifier for the `POSITION` node.
  - *author_id (STRING):* Unique identifier for the author of the content.
  - *text (STRING):* The content associated with the position.
  
- **AGAINST**
  - *id (STRING):* Unique identifier for the `AGAINST` node.
  - *author_id (STRING):* Unique identifier for the author of the content.
  - *text (STRING):* The content associated with the `AGAINST` position.

- **INFAVOR**
  - *id (STRING):* Unique identifier for the `INFAVOR` node.
  - *author_id (STRING):* Unique identifier for the author of the content.
  - *text (STRING):* The content associated with the `INFAVOR` position.
  
- **ENTITY**
  - *value (STRING):* The specific text triplet associated with the entity.

- **WIKI**
  - *id (STRING):* Unique identifier for the `WIKI` node.
  - *wiki_id (STRING):* Unique identifier for the Wikidata entry.
  - *description (STRING):* A detailed description of the Wikidata article or entry.
  - *label (STRING):* The label associated with the Wikidata entry.
  - *tag (STRING):* Tags associated with the Wikidata entry.
  - *wiki_url (STRING):* The URL of the Wikidata entry.

- **CLUSTER**
  - *id (STRING):* Unique identifier for the `CLUSTER` node.
  - *summary (STRING):* A brief summary describing the cluster.
  - *title (STRING):* The title of the cluster.

- **KEYPHRASE**
  - *id (STRING):* Unique identifier for the `KEYPHRASE` node.
  - *keyphrase (STRING):* A key phrase associated with the content.

<br>
Here is the relationships with their propertirs in detail:

- **HAS_POSITION**
  - *Description:* Connects `SUBJECT` nodes to `POSITION` nodes.
  - *Meaning:* Indicates that a specific subject has a particular position.
  - *Properties:* None specified for this relationship.

- **HAS_INFAVOR**
  - *Description:* Connects `POSITION` nodes to `INFAVOR` nodes.
  - *Meaning:* Indicates that a specific position has a particular favor.
  - *Properties:* None specified for this relationship.

- **HAS_AGAINST**
  - *Description:* Connects `POSITION` nodes to `AGAINST` nodes.
  - *Meaning:* Indicates that a specific position has a particular opposition.
  - *Properties:* None specified for this relationship.

- **MENTION**
  - *Description:* Connects `INFAVOR` nodes and `AGAINST` nodes to `ENTITY` nodes.
  - *Meaning:* Indicates that a specific favor or opposition mentions a particular entity (triplet).
  - *Properties:* None specified for this relationship.

- **HAS_POST**
  - *Description:* Connects `CLUSTER` nodes to `AGAINST` nodes and `INFAVOR` nodes.
  - *Meaning:* Indicates that a specific cluster contains a particular favor or opposition.
  - *Properties:* None specified for this relationship.

- **HAS_KEYPHRASE**
  - *Description:* Connects `CLUSTER` nodes to `KEYPHRASE` nodes.
  - *Meaning:* Indicates that a specific cluster contains a particular keyphrase.
  - *Properties:* None specified for this relationship.

- **RELATION**
  - *Description:* Connects `ENTITY` nodes to other `ENTITY` nodes.
  - *Meaning:* Indicates that one entity is related to another entity (triplet).
  - *Properties:* None specified for this relationship.

- **HAS_WIKI_DATA**
  - *Description:* Connects `ENTITY` nodes to `WIKI` nodes.
  - *Meaning:* Indicates that a specific entity has associated Wikidata data.
  - *Properties:* None specified for this relationship.
  
An extra step, **Semantic Enrichment** of KG was also implemented. Specifically, an **Entity Linking System** was used, which links entity mentions in nodes of KG to their corresponding entities in Wikidata. The process is illustrated in the diagram below:

<img src="images/KG_Entity_Linking.png" alt="KG Diagram" width="600">

## Contact
Ioannis Eftsathiou (efstathiou@novelcore.eu)
