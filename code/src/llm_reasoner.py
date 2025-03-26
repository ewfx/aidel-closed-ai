from entity_extractor import entity_extractor_llm


def llm_reasoner(ai_agent_inf,ofac_input=None,graph_input=None,wikidata_input=None):
    llm_input = f"""
             ai agent inferences:{ai_agent_inf}
             OFAC input : {ofac_input}
             Graph database input : {graph_input}
             Wikidata input : {wikidata_input}"""
    output = entity_extractor_llm(llm_input,"prompt_llm2.txt")
    print(output)
    return output


