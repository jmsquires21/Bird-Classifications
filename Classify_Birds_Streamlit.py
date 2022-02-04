# Imports & Setup:
import streamlit as st
import pandas as pd
from pathlib import Path
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import PIL
import PIL.Image
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
tf.random.set_seed(42)



#load models first to improve performance
order_model=tf.keras.models.load_model('saved_models/order_est_v2.h5')
family_model=tf.keras.models.load_model('saved_models/family_v2.h5')
species_model=tf.keras.models.load_model('saved_models/species_t3.h5')

# streamlit code
page = st.selectbox("Navigation", ["Tool", "About"])
if page == "Tool":

    st.title("Classify that Bird! \U0001F426 \U0001F986 \U0001F99C \U0001F424  \U0001F427")
    st.write("")


    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        def load_image_new(path):
            images = Image.open(path)
            new_image=images.resize((128,128))
            color_image=new_image.convert("RGB")
            arrays1 = np.asarray(color_image)
            floaters= arrays1.astype('float32')
            floaters2=floaters/255.0
            floaters3=np.asarray(floaters2)
            floaters4 = floaters3.reshape(1, 128, 128, 3)
            return floaters4

        order_names=["ANSERIFORMES","CAPRIMULGIFORMES","CHARADRIIFORMES","CORACIIFORMES","CUCULIFORMES","GAVIIFORMES","PASSERIFORMES","PELECANIFORMES","PICIFORMES","PODICIPEDIFORMES","PROCELLARIIFORMES","SULIFORMES"]
        proper_ord_names=[i.title() for i in order_names]



        def predict_birds_order_with_prob(file):
            best_model_1 = order_model
            preds1=best_model_1.predict(file)
            preds2=np.argmax(preds1,axis=1)
            max_pred=np.max(preds1)
            max_pred_perc=max_pred*100
            format_pred_perc = "{:.2f}".format(max_pred_perc)
            return format_pred_perc, proper_ord_names[int(preds2)-1]

        family_names=["Albatrosses - Diomedeidae","Auks - Alcidae","Cardinals, Allies - Cardinalidae","Cormorants and Shags - Phalacrocoracidae","Crows, Jays, Magpies - Corvidae","Cuckoos - Cuculidae","Ducks, Geese, Waterfowl - Anatidae","Finches, Euphonias, Allies - Fringillidae","Frigates - Fregatidae","Grebes - Podicipedidae","Gulls, Terns, Skimmers - Laridae","Hummingbirds - Trochilidae","Kingfishers - Alcedinidae","Larks - Alaudidae","Loons - Gaviidae","Mockingbirds and Thrashers - Mimidae","New World Sparrows - Passerellidae","New World Warblers - Parulidae","Nightjars and Allies - Caprimulgidae","Nuthatches - Sittidae","Pelicans - Pelecanidae","Shearwaters and Petrels - Procellariidae","Shrikes - Laniidae","Skuas and Jaegers - Stercorariidae","Starlings - Sturnidae","Swallows - Hirundinidae","Treecreepers - Certhiidae","Troupials and Allies - Icteridae","Tyrant Flycatchers - Tyrannidae","Vireos, Shrike-Babblers, Erpornis - Vireonidae","Wagtails and Pipits - Motacillidae","Waxwings - Bombycillidae","Woodpeckers - Picidae","Wrens - Troglodytidae","Yellow-breasted Chat - Icteriidae"]

        def predict_birds_family_with_prob(file):
            best_model_1f = family_model
            preds1f=best_model_1f.predict(file)
            preds2f=np.argmax(preds1f,axis=1)
            max_predf=np.max(preds1f)
            max_pred_percf=max_predf*100
            format_pred_percf = "{:.2f}".format(max_pred_percf)
            return format_pred_percf, family_names[int(preds2f)-1]


        species_class_names=[
        "001.Black_footed_Albatross - 0.00509",
        "002.Laysan_Albatross - 0.00509",
        "003.Sooty_Albatross - 0.00492",
        "004.Groove_billed_Ani - 0.00509",
        "005.Crested_Auklet - 0.003733",
        "006.Least_Auklet - 0.003478",
        "007.Parakeet_Auklet - 0.004496",
        "008.Rhinoceros_Auklet - 0.004072",
        "009.Brewer_Blackbird - 0.005005",
        "010.Red_winged_Blackbird - 0.00509",
        "011.Rusty_Blackbird - 0.00509",
        "012.Yellow_headed_Blackbird - 0.004751",
        "013.Bobolink - 0.00509",
        "014.Indigo_Bunting - 0.00509",
        "015.Lazuli_Bunting - 0.00492",
        "016.Painted_Bunting - 0.00492",
        "017.Cardinal - 0.004835",
        "018.Spotted_Catbird - 0.003817",
        "019.Gray_Catbird - 0.005005",
        "020.Yellow_breasted_Chat - 0.005005",
        "021.Eastern_Towhee - 0.00509",
        "022.Chuck_will_Widow - 0.004751",
        "023.Brandt_Cormorant - 0.005005",
        "024.Red_faced_Cormorant - 0.004411",
        "025.Pelagic_Cormorant - 0.00509",
        "026.Bronzed_Cowbird - 0.00509",
        "027.Shiny_Cowbird - 0.00509",
        "028.Brown_Creeper - 0.005005",
        "029.American_Crow - 0.00509",
        "030.Fish_Crow - 0.00509",
        "031.Black_billed_Cuckoo - 0.00509",
        "032.Mangrove_Cuckoo - 0.004496",
        "033.Yellow_billed_Cuckoo - 0.005005",
        "034.Gray_crowned_Rosy_Finch - 0.005005",
        "035.Purple_Finch - 0.00509",
        "036.Northern_Flicker - 0.00509",
        "037.Acadian_Flycatcher - 0.005005",
        "038.Great_Crested_Flycatcher - 0.00509",
        "039.Least_Flycatcher - 0.005005",
        "040.Olive_sided_Flycatcher - 0.00509",
        "041.Scissor_tailed_Flycatcher - 0.00509",
        "042.Vermilion_Flycatcher - 0.00509",
        "043.Yellow_bellied_Flycatcher - 0.005005",
        "044.Frigatebird - 0.00509",
        "045.Northern_Fulmar - 0.00509",
        "046.Gadwall - 0.00509",
        "047.American_Goldfinch - 0.00509",
        "048.European_Goldfinch - 0.00509",
        "049.Boat_tailed_Grackle - 0.00509",
        "050.Eared_Grebe - 0.00509",
        "051.Horned_Grebe - 0.00509",
        "052.Pied_billed_Grebe - 0.00509",
        "053.Western_Grebe - 0.00509",
        "054.Blue_Grosbeak - 0.00509",
        "055.Evening_Grosbeak - 0.00509",
        "056.Pine_Grosbeak - 0.00509",
        "057.Rose_breasted_Grosbeak - 0.00509",
        "058.Pigeon_Guillemot - 0.00492",
        "059.California_Gull - 0.00509",
        "060.Glaucous_winged_Gull - 0.005005",
        "061.Heermann_Gull - 0.00509",
        "062.Herring_Gull - 0.00509",
        "063.Ivory_Gull - 0.00509",
        "064.Ring_billed_Gull - 0.00509",
        "065.Slaty_backed_Gull - 0.004242",
        "066.Western_Gull - 0.00509",
        "067.Anna_Hummingbird - 0.00509",
        "068.Ruby_throated_Hummingbird - 0.00509",
        "069.Rufous_Hummingbird - 0.00509",
        "070.Green_Violetear - 0.00509",
        "071.Long_tailed_Jaeger - 0.00509",
        "072.Pomarine_Jaeger - 0.00509",
        "073.Blue_Jay - 0.00509",
        "074.Florida_Jay - 0.00509",
        "075.Green_Jay - 0.004835",
        "076.Dark_eyed_Junco - 0.00509",
        "077.Tropical_Kingbird - 0.00509",
        "078.Gray_Kingbird - 0.005005",
        "079.Belted_Kingfisher - 0.00509",
        "080.Green_Kingfisher - 0.00509",
        "081.Pied_Kingfisher - 0.00509",
        "082.Ringed_Kingfisher - 0.00509",
        "083.White_breasted_Kingfisher - 0.00509",
        "084.Red_legged_Kittiwake - 0.004496",
        "085.Horned_Lark - 0.00509",
        "086.Pacific_Loon - 0.00509",
        "087.Mallard - 0.00509",
        "088.Western_Meadowlark - 0.00509",
        "089.Hooded_Merganser - 0.00509",
        "090.Red_breasted_Merganser - 0.00509",
        "091.Mockingbird - 0.00509",
        "092.Nighthawk - 0.00509",
        "093.Clark_Nutcracker - 0.00509",
        "094.White_breasted_Nuthatch - 0.00509",
        "095.Baltimore_Oriole - 0.00509",
        "096.Hooded_Oriole - 0.00509",
        "097.Orchard_Oriole - 0.005005",
        "098.Scott_Oriole - 0.00509",
        "099.Ovenbird - 0.00509",
        "100.Brown_Pelican - 0.00509",
        "101.White_Pelican - 0.004242",
        "102.Western_Wood_Pewee - 0.00509",
        "103.Sayornis - 0.00509",
        "104.American_Pipit - 0.00509",
        "105.Whip_poor_Will - 0.004157",
        "106.Horned_Puffin - 0.00509",
        "107.Common_Raven - 0.005005",
        "108.White_necked_Raven - 0.00509",
        "109.American_Redstart - 0.00509",
        "110.Geococcyx - 0.00509",
        "111.Loggerhead_Shrike - 0.00509",
        "112.Great_Grey_Shrike - 0.00509",
        "113.Baird_Sparrow - 0.004242",
        "114.Black_throated_Sparrow - 0.00509",
        "115.Brewer_Sparrow - 0.005005",
        "116.Chipping_Sparrow - 0.00509",
        "117.Clay_colored_Sparrow - 0.005005",
        "118.House_Sparrow - 0.00509",
        "119.Field_Sparrow - 0.005005",
        "120.Fox_Sparrow - 0.00509",
        "121.Grasshopper_Sparrow - 0.00509",
        "122.Harris_Sparrow - 0.00509",
        "123.Henslow_Sparrow - 0.00509",
        "124.Le_Conte_Sparrow - 0.005005",
        "125.Lincoln_Sparrow - 0.005005",
        "126.Nelson_Sharp_tailed_Sparrow - 0.005005",
        "127.Savannah_Sparrow - 0.00509",
        "128.Seaside_Sparrow - 0.00509",
        "129.Song_Sparrow - 0.00509",
        "130.Tree_Sparrow - 0.00509",
        "131.Vesper_Sparrow - 0.00509",
        "132.White_crowned_Sparrow - 0.00509",
        "133.White_throated_Sparrow - 0.00509",
        "134.Cape_Glossy_Starling - 0.00509",
        "135.Bank_Swallow - 0.005005",
        "136.Barn_Swallow - 0.00509",
        "137.Cliff_Swallow - 0.00509",
        "138.Tree_Swallow - 0.00509",
        "139.Scarlet_Tanager - 0.00509",
        "140.Summer_Tanager - 0.00509",
        "141.Artic_Tern - 0.00492",
        "142.Black_Tern - 0.00509",
        "143.Caspian_Tern - 0.00509",
        "144.Common_Tern - 0.00509",
        "145.Elegant_Tern - 0.00509",
        "146.Forsters_Tern - 0.00509",
        "147.Least_Tern - 0.00509",
        "148.Green_tailed_Towhee - 0.00509",
        "149.Brown_Thrasher - 0.005005",
        "150.Sage_Thrasher - 0.00509",
        "151.Black_capped_Vireo - 0.004326",
        "152.Blue_headed_Vireo - 0.00509",
        "153.Philadelphia_Vireo - 0.005005",
        "154.Red_eyed_Vireo - 0.00509",
        "155.Warbling_Vireo - 0.00509",
        "156.White_eyed_Vireo - 0.00509",
        "157.Yellow_throated_Vireo - 0.005005",
        "158.Bay_breasted_Warbler - 0.00509",
        "159.Black_and_white_Warbler - 0.00509",
        "160.Black_throated_Blue_Warbler - 0.005005",
        "161.Blue_winged_Warbler - 0.00509",
        "162.Canada_Warbler - 0.00509",
        "163.Cape_May_Warbler - 0.00509",
        "164.Cerulean_Warbler - 0.00509",
        "165.Chestnut_sided_Warbler - 0.00509",
        "166.Golden_winged_Warbler - 0.005005",
        "167.Hooded_Warbler - 0.00509",
        "168.Kentucky_Warbler - 0.005005",
        "169.Magnolia_Warbler - 0.005005",
        "170.Mourning_Warbler - 0.00509",
        "171.Myrtle_Warbler - 0.00509",
        "172.Nashville_Warbler - 0.00509",
        "173.Orange_crowned_Warbler - 0.00509",
        "174.Palm_Warbler - 0.00509",
        "175.Pine_Warbler - 0.00509",
        "176.Prairie_Warbler - 0.00509",
        "177.Prothonotary_Warbler - 0.00509",
        "178.Swainson_Warbler - 0.004751",
        "179.Tennessee_Warbler - 0.005005",
        "180.Wilson_Warbler - 0.00509",
        "181.Worm_eating_Warbler - 0.005005",
        "182.Yellow_Warbler - 0.00509",
        "183.Northern_Waterthrush - 0.00509",
        "184.Louisiana_Waterthrush - 0.00509",
        "185.Bohemian_Waxwing - 0.00509",
        "186.Cedar_Waxwing - 0.00509",
        "187.American_Three_toed_Woodpecker - 0.004242",
        "188.Pileated_Woodpecker - 0.00509",
        "189.Red_bellied_Woodpecker - 0.00509",
        "190.Red_cockaded_Woodpecker - 0.00492",
        "191.Red_headed_Woodpecker - 0.00509",
        "192.Downy_Woodpecker - 0.00509",
        "193.Bewick_Wren - 0.00509",
        "194.Cactus_Wren - 0.00509",
        "195.Carolina_Wren - 0.00509",
        "196.House_Wren - 0.005005",
        "197.Marsh_Wren - 0.00509",
        "198.Rock_Wren - 0.00509",
        "199.Winter_Wren - 0.00509",
        "200.Common_Yellowthroat - 0.00509"
        ]

        def predict_birds_species_with_prob(file):
            best_model_1f = species_model
            preds1f=best_model_1f.predict(file)
            preds2f=np.argmax(preds1f,axis=1)
            max_predf=np.max(preds1f)
            max_pred_percf=max_predf*100
            format_pred_percf = "{:.2f}".format(max_pred_percf)
            return format_pred_percf, species_class_names[int(preds2f)-1]


        image_new=load_image_new(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.image(image_new, caption='Processed Image.', use_column_width=True)
        st.write("")
        st.write("Classifying the image...")
        #label = predict_birds_order(image_new)
        label_of_probability= predict_birds_order_with_prob(image_new)
        label_of_probability_family= predict_birds_family_with_prob(image_new)
        label_of_probability_species= predict_birds_species_with_prob(image_new)


        st.write('**Order**')
        st.write(f'The model is {label_of_probability[0]}% sure that this bird belongs to the **{label_of_probability[1]}** order!')

        st.write('**Family**')
        st.write(f'The model is {label_of_probability_family[0]}% sure that this bird belongs to the **{label_of_probability_family[1]}** family!')

        st.write('**Species**')
        st.write(f'The model is {label_of_probability_species[0]}% sure that this bird belongs to the **{label_of_probability_species[1].split("-", 1)[0].split(".", 1)[1].replace("_", " ")}** species!')
        #st.write(f'This bird belongs to the {label} order!')

        st.write("**About this bird's predicted order**")
        if label_of_probability[1]=='Anseriformes':
            st.write("Anseriformes is an order of birds also known as waterfowl that comprises about 180 living species of birds in three families: Anhimidae, Anseranatidae, and Anatidae, the largest family, which includes over 170 species of waterfowl, among them the ducks, geese, and swans.")
        elif label_of_probability[1]=='Caprimulgiformes':
            st.write("Caprimulgiformes is an order of birds also known as nightjars, any of about 120 species of soft-plumaged birds, the major groups of which are called nightjars, nighthawks, potoos, frogmouths, and owlet-frogmouths.")
        elif label_of_probability[1]=='Charadriiformes':
            st.write("Charadriiformes is an order of birds also known as shorebirds, and they are comprised of waders, gulls, and auks. They include approximately 390 species and are found all around the world.")

        elif label_of_probability[1]=='Coraciiformes':
            st.write("The Coraciiformes are a group of usually colourful birds including the kingfishers, the bee-eaters, the rollers, the motmots, and the todies. They generally have syndactyly, with three forward-pointing toes, though in many kingfishers one of these is missing.")
        elif label_of_probability[1]=='Gaviiformes':
            st.write("Gaviiformes is an order of aquatic birds containing the loons or divers and their closest extinct relatives. Modern gaviiformes are found in many parts of North America and northern Eurasia, though prehistoric species were more widespread.")
        elif label_of_probability[1]=='Passeriformes':
            st.write("Passeriformes includes more than half of all bird species. Sometimes known as perching birds or songbirds, passerines are distinguished from other orders by the arrangement of their toes, which facilitates perching.")
        elif label_of_probability[1]=='Pelecaniformes':
            st.write("The Pelecaniformes are an order of medium-sized and large waterbirds found all around the world. Traditionally defined, they include birds that have four toes webbed.")

        elif label_of_probability[1]=='Piciformes':
            st.write("Nine families of largely arboreal birds make up Piciformes, the best-known of them being the Picidae, which includes the woodpeckers and close relatives.")
        elif label_of_probability[1]=='Podicipediformes':
            st.write("Grebes are aquatic diving birds in the order Podicipediformes. Grebes are widely distributed birds of freshwater, with some species also occurring in marine habitats during migration and winter.")
        elif label_of_probability[1]=='Procellariiformes':
            st.write("Procellariiformes is an order of seabirds that comprises four families: the albatrosses, the petrels and shearwaters, and two families of storm petrels.")
        elif label_of_probability[1]=='Suliformes':
            st.write("Suliformes are an order of birds that consist of Frigatebirds, Gannets, Anhingas, Cormorants and Shags.")
        else:
            st.write("")
        st.write("")
        st.write("")
        st.write("*Fun bird facts provided by [The Cornell Lab](https://www.allaboutbirds.org/guide/browse/taxonomy) and [Wikipedia](https://en.wikipedia.org/wiki/Main_Page)  *")

# Page to read about the project
elif page == "About":
    st.title("")
    st.write("")
    st.markdown("")

    def read_markdown_file(markdown_file):
        return Path(markdown_file).read_text()

    intro_markdown = read_markdown_file("about_streamlit.md")
    st.markdown(intro_markdown, unsafe_allow_html=True)
