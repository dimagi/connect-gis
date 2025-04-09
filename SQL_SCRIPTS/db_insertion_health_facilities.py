import json
from sqlalchemy import create_engine, Column, Integer, String, JSON, text
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.inspection import inspect
from geoalchemy2 import Geometry

# Database connection
DATABASE_URI = "postgresql://username:password@localhost:5432/your_database"

# Initialize SQLAlchemy
engine = create_engine(DATABASE_URI)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()


# Define the HealthFacility model
class HealthFacility(Base):
    __tablename__ = 'health_facilities'

    id = Column(Integer, primary_key=True)
    country_name = Column(String)
    country_code = Column(String)
    state_name = Column(String)
    state_code = Column(String)
    lga_name = Column(String)
    lga_code = Column(String)
    ward_name = Column(String)
    ward_code = Column(String)
    name = Column(String)
    global_id = Column(String)
    source = Column(String)
    source_date = Column(String)
    properties = Column(JSON)  # Stores the nested properties
    sub_type = Column(String)
    geom = Column(Geometry(geometry_type='POINT', srid=4326))


# Create table if not exists
def create_table():
    inspector = inspect(engine)

    # Check if table exists
    if not inspector.has_table('health_facilities'):
        # Enable PostGIS if not already enabled
        session.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        session.commit()

        # Create the table
        Base.metadata.create_all(engine)
        print("Table 'health_facilities' created successfully")
    else:
        print("Table 'health_facilities' already exists")


# Process and insert GeoJSON data
def insert_geojson_data(geojson_file):
    with open(geojson_file) as f:
        data = json.load(f)

    features = data['features']
    print(f"Found {len(features)} health facilities to insert")

    for feature in features:
        props = feature['properties']
        geometry = feature['geometry']

        facility = HealthFacility(
            country_name=props.get('country_name'),
            country_code=props.get('country_code'),
            state_name=props.get('state_name'),
            state_code=props.get('state_code'),
            lga_name=props.get('lga_name'),
            lga_code=props.get('lga_code'),
            ward_name=props.get('ward_name'),
            ward_code=props.get('ward_code'),
            name=props.get('name'),
            global_id=props.get('global_id'),
            source=props.get('source'),
            source_date=props.get('source_date'),
            properties=props.get('properties', {}),
            sub_type=props.get('sub_type'),
            geom=f"SRID=4326;POINT({geometry['coordinates'][0]} {geometry['coordinates'][1]})"
        )

        session.add(facility)

    session.commit()
    print(f"Successfully inserted {len(features)} health facilities")


if __name__ == "__main__":
    GEOJSON_FILE = "../static/nigeria_health_facilities.geojson"

    try:
        create_table()
        insert_geojson_data(GEOJSON_FILE)

    except Exception as e:
        print(f"Error: {e}")
        session.rollback()
    finally:
        session.close()