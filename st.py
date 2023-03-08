import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from catboost import CatBoostRegressor
from streamlit_shap import st_shap
import shap



st.set_page_config(page_title="Rental Price Prediction",layout="centered",page_icon="🏠") 
st.title('U.S Rental Price Prediction 🏠')

state= st.selectbox(
'State',
('co', 'ne', 'wv', 'fl', 'ms', 'ca', 'ny', 'dc', 'ks', 'nd', 'nc',
   'ct', 'ok', 'nm', 'tn', 'or', 'az', 'il', 'sc', 'al', 'tx', 'sd',
   'la', 'wa', 'pa', 'oh', 'ar', 'ky', 'in', 'mn', 'ga', 'md', 'ia',
   'va', 'mi', 'ma', 'nj', 'ut', 'id', 'de', 'mo', 'ak', 'wi', 'nv',
   'mt', 'ri', 'nh', 'vt', 'hi', 'wy', 'me'))
region = st.selectbox('Region',
('boulder', 'lincoln', 'morgantown', 'south florida', 'hattiesburg',
   'san diego', 'albany', 'washington, DC', 'kansas city, MO',
   'bismarck', 'greensboro', 'fresno / madera', 'space coast',
   'new haven', 'long island', 'oklahoma city', 'albuquerque',
   'tri-cities', 'medford-ashland', 'charlotte', 'phoenix',
   'inland empire', 'champaign urbana', 'myrtle beach',
   'huntsville / decatur', 'winston-salem',
   'killeen / temple / ft hood', 'lakeland', 'sioux falls / SE SD',
   'denver', 'baton rouge', 'skagit / island / SJI', 'gainesville',
   'stockton', 'poconos', 'st louis, MO', 'odessa / midland', 'yuma',
   'panama city', 'grand forks', 'memphis', 'topeka', 'cincinnati',
   'pensacola', 'tuscaloosa', 'little rock', 'rochester',
   'ft myers / SW florida', 'daytona beach', 'mobile', 'louisville',
   'jackson', 'modesto', 'north dakota', 'sacramento', 'syracuse',
   'cleveland', 'evansville', 'minneapolis / st paul', 'wenatchee',
   'seattle-tacoma', 'glens falls', 'baltimore',
   'omaha / council bluffs', 'asheville', 'eastern CT',
   'fredericksburg', 'lansing', 'western massachusetts',
   'savannah / hinesville', 'rapid city / west SD', 'manhattan',
   'annapolis', 'columbus', 'fort collins / north CO', 'western KY',
   'flagstaff / sedona', 'south jersey', 'new orleans',
   'provo / orem', 'wichita', 'boise', 'delaware', 'akron / canton',
   'jacksonville', 'bellingham', 'saginaw-midland-baycity',
   'springfield', 'salem', 'athens', 'san marcos', 'port huron',
   'tallahassee', 'college station', 'anchorage / mat-su',
   'south coast', 'richmond', 'hilton head', 'raleigh / durham / CH',
   'wilmington', 'kenosha-racine', 'nashville', 'florida keys',
   'detroit metro', 'dallas / fort worth', 'yakima', 'visalia-tulare',
   'madison', 'columbia', 'norfolk / hampton roads', 'battle creek',
   'gulfport / biloxi', 'fargo / moorhead', 'grand rapids', 'merced',
   'bowling green', 'pittsburgh', 'ventura county', 'orlando', 'ames',
   'ogden-clearfield', 'santa barbara', 'peoria', 'chattanooga',
   'SF bay area', 'lafayette', 'las vegas', 'lubbock', 'indianapolis',
   'chicago', 'monterey bay', 'houston', 'flint', 'corpus christi',
   'waco', 'sarasota-bradenton', 'austin', 'el paso', 'janesville',
   'kalispell', 'knoxville', 'dayton / springfield', 'san antonio',
   'toledo', 'corvallis/albany', 'pullman / moscow', 'augusta',
   'rhode island', 'st augustine', 'fayetteville', 'frederick',
   'columbia / jeff city', 'sioux city', 'abilene', 'lexington',
   'hartford', 'hudson valley', 'charleston', 'orange county',
   'tyler / east TX', 'san luis obispo', 'houma', 'milwaukee',
   'charlottesville', 'shreveport', 'heartland florida',
   'north mississippi', 'joplin', 'mcallen / edinburg', 'bozeman',
   'erie', "spokane / coeur d'alene", 'new hampshire',
   'bloomington-normal', 'birmingham', 'pueblo', 'los angeles',
   'wichita falls', 'tulsa', 'moses lake', 'lehigh valley', 'vermont',
   'salt lake city', 'des moines', 'philadelphia', 'north jersey',
   'kennewick-pasco-richland', 'youngstown', 'auburn', 'eastern NC',
   'southern maryland', 'waterloo / cedar falls', 'hawaii',
   'portland', 'atlanta', 'tampa bay area', 'lancaster', 'buffalo',
   'chillicothe', 'bakersfield', 'quad cities, IA/IL', 'texoma',
   'treasure coast', 'ann arbor', 'jersey shore', 'york',
   'sierra vista', 'plattsburgh-adirondacks', 'greenville / upstate',
   'southwest michigan', 'northern michigan', 'eastern shore',
   'amarillo', 'eugene', 'eastern kentucky', 'state college',
   'macon / warner robins', 'northern panhandle', 'reno / tahoe',
   'montgomery', 'harrisonburg', 'muncie / anderson', 'western slope',
   'st cloud', 'scranton / wilkes-barre', 'colorado springs', 'chico',
   'eastern panhandle', 'high rockies', 'victoria', 'redding',
   'bloomington', 'okaloosa / walton', 'decatur', 'duluth / superior',
   'lynchburg', 'st george', 'iowa city', 'monroe',
   'beaumont / port arthur', 'gadsden-anniston', 'boston',
   'stillwater', 'galveston', 'central NJ', 'palm springs',
   'south bend / michiana', 'lawrence', 'logan', 'valdosta',
   'southeast alaska', 'cumberland valley', 'st joseph', 'muskegon',
   'wyoming', 'catskills', 'florence', 'worcester / central MA',
   'scottsbluff / panhandle', 'show low', 'elko', 'dothan',
   'deep east texas', 'maine', 'finger lakes', 'harrisburg',
   'mansfield', 'mohave county', 'east oregon', 'rockford',
   'missoula', 'winchester', 'appleton-oshkosh-FDL', 'watertown',
   'reading', 'tucson', 'brownsville', 'fort wayne',
   'utica-rome-oneida', 'jonesboro', 'ithaca', 'kenai peninsula',
   'la crosse', 'binghamton', 'ocala', 'cape cod / islands',
   'holland', 'roanoke', 'bend', 'oregon coast', 'the thumb',
   'cookeville', 'east idaho', 'huntington-ashland', 'kalamazoo',
   'mankato', 'lake of the ozarks', 'zanesville / cambridge',
   'cedar rapids', 'lewiston / clarkston', 'hanford-corcoran',
   'clarksville', 'las cruces', 'humboldt county', 'klamath falls',
   'santa fe / taos', 'southwest MN', 'altoona-johnstown',
   'northwest GA', 'meridian', 'imperial county', 'northwest KS',
   'northwest OK', 'fairbanks', 'florence / muscle shoals',
   'southern illinois', 'williamsport', 'bemidji', 'new york city',
   'sheboygan', 'lawton', 'green bay', 'brainerd', 'western maryland',
   'brunswick', 'sandusky', 'billings', 'outer banks',
   'parkersburg-marietta', 'lake charles', 'san angelo',
   'mattoon-charleston', 'lafayette / west lafayette', 'yuba-sutter',
   'danville', 'tuscarawas co', 'eau claire', 'laredo',
   'mendocino county', 'helena', 'twin tiers NY/PA', 'prescott',
   'north central FL', 'potsdam-canton-massena', 'gold country',
   'salina', 'terre haute', 'southwest VA', 'elmira-corning',
   'eastern montana', 'clovis / portales', 'santa maria',
   'statesboro', 'southeast KS', 'del rio / eagle pass',
   'central michigan', 'roseburg', 'wausau', 'hickory / lenoir',
   'roswell / carlsbad', 'texarkana', 'new river valley', 'dubuque',
   'southern WV', 'northwest CT', 'fort dodge', 'meadville',
   'siskiyou county', 'south dakota', 'farmington',
   'olympic peninsula', 'southeast missouri', 'oneonta', 'fort smith',
   'kokomo', 'southeast IA', 'upper peninsula', 'la salle co',
   'great falls', 'eastern CO', 'butte', 'northeast SD',
   'lima / findlay', 'owensboro', 'twin falls', 'central louisiana',
   'ashtabula', 'northern WI', 'grand island', 'mason city',
   'pierre / central SD', 'chautauqua', 'kirksville', 'north platte',
   'western IL', 'southwest TX', 'boone', 'southwest MS',
   'southwest KS'))

type = st.selectbox(
'Type',
('apartment', 'duplex', 'townhouse', 'condo', 'cottage/cabin',
   'house', 'manufactured', 'loft', 'flat', 'in-law'))
SquareFeet = st.slider('SquareFeet',200,3500,1000,key=12)
beds = st.slider('Beds',1,5,2,key=5)
baths	= st.slider('Baths',1,5,2,key=6)
comes_furnished	= st.slider('Furnished',0,1,0,key=9)
parking_options	= st.selectbox(
'Parking Options?',
('no parking', 'off-street parking', 'attached garage', 'carport',
   'detached garage', 'street parking', 'valet parking'))
laundry_options	= st.selectbox(
'Laundry Options?',
('no laundry on site', 'w/d in unit', 'w/d hookups',
   'laundry on site', 'laundry in bldg'))
lat	= st.number_input('Latitude')
long	= st.number_input('Longtitude')
      
data = {'state': state,
    'region': region,
   'type': type,
   'sqfeet': SquareFeet,
   'beds': beds,
   'baths': baths,
   'comes_furnished': comes_furnished,
   'parking_options':parking_options ,
   'laundry_options': laundry_options,
   'lat': lat,
   'long': long,
   }
features = pd.DataFrame(data, index=[0])
      

categoricalcolumns = features.select_dtypes(include=["object"]).columns.tolist()

from catboost import Pool
import pickle
pool1=Pool(data=features,cat_features=categoricalcolumns)

with open('catboost_model-2.pickle', 'rb') as f:
    model = pickle.load(f)
prediction=model.predict(pool1)
pred=prediction[0]

st.set_option('deprecation.showPyplotGlobalUse', False)

if st.button('Predict Rental Price'):
   st.header("The predicted rental price is " + " $" + str(round(pred,2)))
   explainer = shap.TreeExplainer(model)
   shap_values = explainer.shap_values(pool1)
   st_shap(shap.plots.force(explainer.expected_value,shap_values[0],feature_names=features.columns))
   shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0],feature_names=features.columns)
   st.pyplot(bbox_inches='tight')







