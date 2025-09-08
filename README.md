# EU_ORBIS

This repository contains the implementation of EU project: ORBIS

**Abstract:** ORBIS addresses the disconnects between ambitious ideas and collective actions at a large socio-technical scale. It responds to the profound lack of dialogue between citizenship and policy making institutions by providing a theoretically sound and highly pragmatic socio-technical solution to enable the transition to a more inclusive, transparent and trustful Deliberative Democracy in Europe. The project shapes and supports new democratic models that are developed through deliberative democracy processes; it follows a socioconstructive approach in which deliberative democracy is not a theory which prescribes new democratic practices and models, but rather the process through which we can collectively imagine and realize them. ORBIS provides new ways to understand and facilitate the emergence of new participatory democracy models, together with the mechanisms to scale them up and consolidate them at institutional level. It delivers: (i) a sound methodology for deliberative participation and co-creation at scale; (ii) novel AI-enhanced tools for deliberative participation across diverse settings; (iii) a novel socio-technical approach that augments the articulation between deliberative processes and representative institutions in liberal democracies; (iv) new evidence-based democratic models that emerge from the application of citizen deliberation processes; (v) demonstrated measurable impact of such innovations in real-world settings. The project builds on cutting-edge AI tools and technologies to develop a sustainable digital solution, and bridges theories and technological solutions from the fields of political and social science, social innovation, Artificial Intelligence, argumentation and digital democracy. The achievement of the project’s goal is validated through six use cases addressing contemporary issues at different scales and settings, experimenting with different civic participation and deliberation models, and involving diverse types of stakeholders.

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

## Implementations
The repository contains two implementations, which can be found in the `BcauseOrbis` and `PolisOrbis` folders.

### Pipeline Overview
Each implementation follows a 4-step pipeline:
1. **Data Download**: Retrieves data from respective platforms (BCAUSE/Pol.is)
2. **Clustering**: Groups similar feedback using Fuzzy C-Means clustering
3. **Knowledge Graph Construction**: Creates semantic knowledge graphs with entity linking
4. **Policy Recommendations**: Generates actionable policy recommendations from cluster insights