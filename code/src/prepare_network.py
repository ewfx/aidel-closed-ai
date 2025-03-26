from neo4j import GraphDatabase

db_uri = "bolt://localhost:7687"
db_user = "neo4j"
db_password = "password"
driver = GraphDatabase.driver(db_uri, auth=(db_user, db_password))


def create_full_text_index():
    """Creates a full-text index on Entity, Officer, and Address names."""
    with driver.session() as session:
        session.run("CREATE FULLTEXT INDEX entity_name_index FOR (e:Entity) ON EACH [e.name]")
        session.run("CREATE FULLTEXT INDEX officer_name_index FOR (o:Officer) ON EACH [o.name]")
        session.run("CREATE FULLTEXT INDEX intermediary_name_index FOR (i:Intermediary) ON EACH [i.name]")
        session.run("CREATE FULLTEXT INDEX address_name_index FOR (a:Address) ON EACH [a.address]")

create_full_text_index()