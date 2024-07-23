import random
import pandas as pd

# Defining training, validation, and test sets
train_data = [
    ("Entry: Enter your age bellow", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Entry: Enter your date of birth", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("The content on this site is only suited for a mature audience", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Mature Content, viewer discretion is advised", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("You must be over-age to access this site", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Are you overage?", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Ensure you are over the required legal drinking age to access this site.", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Are you over-age?", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Only enter if you're over the legal drinking age in your country", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("This site is intended for adults only.", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Enter your D.O.B: ", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Enter your DOB: ", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("You must be 18+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Age verification required", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Must be 21 or older", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Only for adults", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("18+ only", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Verify your age", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Enter your birthdate", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Are you over 18?", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("This site is for 18+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Age restricted content", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Must be of legal drinking age", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Adults only, please", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Confirm your age", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("You must be 21+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Restricted to 18+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Verify that you are over 18", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("This content is for adults only", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Please confirm your age", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Must be 18 years or older", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Age restriction applies", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("18+ verification needed", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("You need to be 18 or older", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Check your age", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Age check required", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Adult supervision required", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Not suitable for minors", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Enter your age", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Legal age verification", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("18+ age check", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Verify age to continue", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Adults only section", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("You must be of legal age", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Age-restricted access", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Age gate - Are you 18+?", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("This is an adult site", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Please verify you are 21+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Must be 21 to enter", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Age confirmation needed", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Restricted to adults", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("This site requires age verification", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Verify you are an adult", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("This site is 18+ only", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("You must confirm your age", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Age gate: 18+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Adult content warning", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("You need to be 21+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Age verification mandatory", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Confirm you are 18+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Restricted content - 18+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Age limit: 18+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Enter your birth year", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Must be 18 to proceed", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Only accessible by adults", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Please verify your birthdate", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Must be of legal age to view", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Restricted to legal age users", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Confirm your birth year", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("You must be an adult", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Age check: 21+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Restricted area: 18+ only", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("You must be 18 or older to continue", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("This content is for users 18+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Confirm age to access", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Age verification: 18+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Verify you are 18 or older", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("You must verify your age to continue", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("This site is restricted to 18+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Please confirm you are of legal age", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Age gate: 21+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Confirm your age to proceed", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("This is an age-restricted site", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Please verify you are over 18", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("18+ age restriction", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Verify your age: 18+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("This site is for adults aged 18+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Enter your birthdate to proceed", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Must be 18 to access", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Age check: 18+ only", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Restricted content for adults", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("You must be of legal drinking age to enter", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Confirm age: 18+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("You must verify your age: 18+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("This content is age-restricted to 18+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Verify your age to access", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Please confirm your age: 18+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("This section is for adults only", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("18+ content ahead", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Confirm you are over 18", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Restricted access: 18+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Verify your age to continue: 18+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("You must be an adult to enter", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Please verify you are 18+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Restricted site for 18+ users", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("This page is for adults only", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Age confirmation: 18+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("You need to be 18+ to view", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("This content is restricted to adults", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Please confirm your birthdate", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("You must be 18+ to access this content", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Enter your age to continue", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Restricted to legal age", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Must be 21 or older to enter", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("This content is for legal age only", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Age check: must be 18+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Please verify your age: 21+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("You must be of legal age: 18+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Confirm you are of legal drinking age", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Enter birth year to verify age", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Age-restricted content: 18+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Verify age: must be 18+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("This site is for adults over 18", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("You need to confirm your age", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Age gate: please verify age", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Enter birthdate to confirm age", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("This page requires age verification", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Verify you are of legal age", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Age verification: must be 18+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("This content is for 18+ users only", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Confirm you are 18 or older", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Please enter your birthdate", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Verify your age to proceed", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Restricted content: age 18+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("You must be 18+ to view this", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Confirm age: must be 18+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("This site requires age confirmation", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Please enter your birth year", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Restricted to users over 18", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("This site is age-restricted: 18+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("You need to be over 18 to enter", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Age verification required: 18+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Confirm your age: 21+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("You must be 21+ to access this content", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Verify you are over 18 to continue", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("This content is restricted to 21+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Please confirm you are 18 or older", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Enter birthdate to verify", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("You must be of legal age to access", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Confirm you are of legal age to proceed", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Age-restricted site: 18+ only", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("You need to be 18+ to proceed", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("This is an age-restricted page", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Verify you are 21+ to access", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Please verify your birth year", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("This content is for 18+ viewers", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Confirm you are an adult to enter", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Age confirmation required: 18+", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Verify age to access content", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("This is an adult content site", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Please verify your age to continue", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("You must confirm your birthdate", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Are you 13+?",{"cats": {"AGE_VERIFICATION": 1.0}}),
    
    # Non age Verification examples
    ("This website is suitable for all ages.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Welcome to our website!", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Please login to access your account.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("This website is intended for general audiences.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("You can access our website from anywhere in the world.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Our website is open to everyone.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Please register to access our services.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("This website is designed for educational purposes.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("You can use our website for free.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Our website is available in multiple languages.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Please contact us for more information.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("This website is updated regularly.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("You can search our website for specific content.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Our website is secure and trustworthy.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Please read our terms and conditions.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("This website is intended for entertainment purposes only.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("You can access our website 24/7.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Our website is user-friendly and easy to navigate.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Please provide your feedback to help us improve.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("This website is designed for informational purposes only.", {"cats": {"AGE_VERIFICATION": 0.0}}),

    # Random unrelated data
    ("Your cart is empty Shop our products", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("RARE BOTTLES BUNDLES WHISKEY TEQUILA VODKA GIN COGNAC & BRANDY LIQUEUR & OTHERS WINE GIFTS Need help? shoponlineliquor@gmail.com Follow Us", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Facebook Instagram TikTok Back RARE BOTTLES Blanton's Pappy Van Winkle W.L. Weller Elmer T. Lee Whistlepig The Boss Hog Colonel", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("E.H. Taylor Rock Hill Farms Yamazaki Hibiki Michters Back BUNDLES Rare Bundles Bourbon Bundles Whiskey Bundles Tequila Bundles Vodka Bundles", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("BOX AND BATTERIES", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("passm", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("passwordm", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("datem", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Accomodation for adults", {"cats": {"AGE_VERIFICATION": 0.0}}),

    ("B&Bsm", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("2 adults, 3 children and no pets", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("4 Adults and one child", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Check-out date adults and children, looking for flights", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Save 20%", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Save up to 18%", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Benefit from up to 50\% discounts", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Prices are now 18$", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Updated price is 21€", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Check-in check-out", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Outdoorsm", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Over 20+ new items", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Over 1000+ articles", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("10000+ articlesm", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Over the bridge", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("under 59.99$", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("price: 49.99$", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("99.99€ for leather items", {"cats": {"AGE_VERIFICATION": 0.0}}),
    (" -20%", {"cats": {"AGE_VERIFICATION": 0.0}}),
    (" -30%", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("item under 5€", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Positive reviews: 4,293,112 users have liked this post", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("-33% 49,99€ 33,49€m ", {"cats": {"AGE_VERIFICATION": 0.0}}),
    (" -67%m ", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("-30% 24,99€ 17,49€ Live -20% 24,50€ 19,60€ 7,99€m  ", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("-43%m ", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("-25% 49,99€ 37,49€m ", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Under 12€", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Under 13$", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("-70\% in all retail prices", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("18 days", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("21 and counting", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Video Teenage 188K 85% 3 weeks ago ", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Orientation All Gay Transgender ", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("All gay category", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Visually Impaired", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("May cause epilepsy", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Liana_Li Not interested 18:56m ", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Next year", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("2 years", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("4 years ago", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("7 months is the limit", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Get 1 month of free access", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Get exclusive content for free", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Get free videos for 7 days", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("New promotion", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("2xm", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("1,000,000+ articles", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Shoes under $50", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Personal Care under $25", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Jeans under $50 ", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("For teens", {"cats": {"AGE_VERIFICATION": 0.0}}),   
    ("For Adults", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("For men", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("women and children section", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("enter email and password", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("id and password required", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("username and password", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Content warning", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Third person shooter", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Best prices in town, with mad discounts up to -50\%!", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("30\% sales now available!", {"cats": {"AGE_VERIFICATION": 0.0}}), 
    ("Entry: Consider entering a short position ar", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Entry: Enter your name and address", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Entry: Postal code required", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Entry: Clearly state your intentions", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Consider leaving a review bellow", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Consider entering a short position", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Consider placing a put option", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Consider placing a market order", {"cats": {"AGE_VERIFICATION": 0.0}}),

    ("                            adresses", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Contact telephone number : +33 657 55 23 43", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Your cart is empty Shop our products", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The quick brown fox jumps over the lazy dog.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("She sells seashells by the seashore.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("He has a pen and a notebook for school.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The sky is clear and blue today.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("I enjoy reading books in my spare time.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("This recipe requires two cups of flour.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("They are planning a trip to the mountains.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The meeting will start at 10 AM.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("She loves to paint landscapes and portraits.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("He bought a new car last week.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The weather forecast predicts rain tomorrow.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("She listens to music while jogging.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The movie starts at 7 PM tonight.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("He is learning to play the guitar.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The park is a great place for a picnic.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("She works in a downtown office building.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("He enjoys cooking Italian dishes.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The children are playing in the playground.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("She writes in her journal every evening.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("They are watching a new TV series together.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Hello, how are you?", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The sky is blue.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("I love programming.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("This is a test phrase.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Have a great day!", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Welcome to our website.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The quick brown fox jumps over the lazy dog.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("What time is it?", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Can you help me with this?", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("It's a beautiful day outside.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Please call me later.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The weather is nice today.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("I enjoy reading books.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("This is not related to age verification.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("How's it going?", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Let's go for a walk.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("She sells seashells by the seashore.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("This is an example sentence.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("What is your name?", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("I need to finish my homework.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Let's meet at the park.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Can you believe it?", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The cat sat on the mat.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Where did you go?", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("I like to play soccer.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("She is reading a book.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("What do you think?", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("This is a sample text.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Can you send me the file?", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The sun is shining.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("It's raining outside.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Where is the nearest store?", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("I need to buy groceries.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The movie was great.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("I have a meeting at 3 PM.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The flowers are blooming.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("What did you have for lunch?", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Can you hear that sound?", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The book is on the table.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("She is watching TV.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("I need to go now.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("What are you doing?", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("He is playing the guitar.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The car is parked outside.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Let's have a coffee break.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Can you see the stars?", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The phone is ringing.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("I forgot my keys.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("She is wearing a red dress.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("What is your favorite color?", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The door is open.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("I am writing a letter.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Let's go to the beach.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Can you give me a hand?", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The meeting was productive.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("She has a beautiful smile.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("I am learning to cook.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("What is your favorite book?", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The dog is barking.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("I need to charge my phone.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("She is a talented artist.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The laptop is on the desk.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Can you pass the salt?", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The cake tastes delicious.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("I am feeling tired.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("She loves to dance.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The train is arriving.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Let's watch a movie.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("He is writing a novel.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The bike is in the garage.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Can you close the window?", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The tree is tall.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("I need to water the plants.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("She is singing a song.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The coffee is hot.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Sale price $311.99 Regular price $399.99", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("New price $239.99!", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Regular pricem", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Passwordm ; email ; password", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Clear Close", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Year 750ml W.L. Weller Reviews.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Rare Bundlesm ", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Let's go for a run.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The clock is ticking.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("I am baking cookies.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("She has a new haircut.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The plane is landing.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Can you open the door?", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The chair is broken.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("I need to send an email.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("She is wearing a hat.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The road is long.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Let's go for dinner.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The baby is sleeping.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("I am reading the news.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("She is writing a letter.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The music is playing.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("I need to clean the house.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("He is driving the car.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The ice cream is melting.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Let's have a picnic.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The stars are bright.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("I am going for a walk.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("She is painting a picture.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The phone needs charging.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Can you help with the dishes?", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The water is cold.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("I need to find my glasses.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("She is listening to music.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The garden is beautiful.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Let's visit the museum.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The street is busy.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("I am working from home.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("She is knitting a sweater.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The lamp is on.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Can you write it down?", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The bridge is old.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("I am learning Spanish.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("She is making a call.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The food is ready.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Let's play a game.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The mountain is high.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("I am drawing a picture.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("She is feeding the cat.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The bus is late.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Can you hear the birds?", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The river is calm.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("I need to wash the car.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("She is reading a magazine.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The street is quiet.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Let's go to the zoo.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The city is crowded.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("I am fixing the sink.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("She is planting flowers.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The roof is leaking.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Can you hold this for me?", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The bakery is open.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("I need to buy a gift.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("She is riding a bike.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The snow is falling.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Let's build a snowman.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The lake is frozen.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("I am learning to swim.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("She is making dinner.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The sky is clear.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Can you tell me the time?", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The book is interesting.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("I am making a presentation.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("She is drinking coffee.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The bed is comfortable.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Let's go shopping.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The beach is crowded.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("I am decorating the house.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("She is playing chess.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The car needs gas.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Can you see the moon?", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The grass is green.", {"cats": {"AGE_VERIFICATION": 0.0}}),
]
random.shuffle(train_data) # Ensures randomisation whenver called
train_df = pd.DataFrame(train_data, columns=['text', 'label'])
x_train = train_df['text'].tolist()
y_train = train_df['label'].apply(lambda x: x['cats']['AGE_VERIFICATION']).tolist()

validation_data = [
    ("This website requires age verification to proceed. Are you over 18?", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("You must be at least 18 years old to participate. If not please exit.", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("This website requires parental confirmation before you proceed.", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Are you 13+ ?", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Enter your DOB bellow: ", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Enter your birth date bellow: ", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("You must be 17+ to enter.", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Content restricted to those 16 and older.", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Access limited to 21+ only.", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Please verify you are 13 or older.", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("This site is for users 19+.", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Only accessible if you are 20+.", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Confirm your age: must be 15+.", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Enter your birth year to proceed.", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("This section is for 14+ only.", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Age verification: 13+ required.", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Must be at least 18 to view.", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Access for those aged 22+.", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Verify you are over 19.", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Restricted to users 23+.", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Only for individuals 24 and above.", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Please confirm you are 25 or older.", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("You need to be 26+ to continue.", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Enter your age: must be 27+.", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("This site is restricted to those 28 and over.", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Only available to users 29+.", {"cats": {"AGE_VERIFICATION": 1.0}}),
    ("Verify age: must be 30+.", {"cats": {"AGE_VERIFICATION": 1.0}}),

    ("I have 2 cats and 1 dog.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("There are 7 days in a week.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("I will arrive at 5 PM.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("She bought 3 apples.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The book costs $15.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("There are 50 states in the USA.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("He scored 100 points in the game.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The building has 12 floors.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("I need 4 eggs for the recipe.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("She is 30 years old.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The phone number is 555-1234.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("He ran 10 miles yesterday.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("I need to wake up at 6 AM.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("There are 8 planets in the solar system.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The movie starts at 7:30 PM.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("She bought 5 books.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The event is on July 4th.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("There are 24 hours in a day.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("He needs 3 hours to finish the task.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("She has 2 brothers.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The car's speed limit is 60 mph.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("I have visited 10 countries.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("There are 4 seasons in a year.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The concert is on October 12th.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The store opens at 9 AM.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("He paid $20 for the ticket.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The meeting is at 3 PM.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("There are 11 players on a soccer team.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("The temperature is 75 degrees.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("I have 5 assignments to complete.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Our website is user-friendly!", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Please provide your feedback to: +33 675 456 321", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("This website is designed for entertainment purposes only.", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("e-cigarettes", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("enter your licence numeber: ", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("Under 5€", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("12\% discount applied", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("blondm", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("131K likes on this post", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("popular mature videos", {"cats": {"AGE_VERIFICATION": 0.0}}),
    ("14 daysm", {"cats": {"AGE_VERIFICATION": 0.0}}),
]
random.shuffle(validation_data)
valid_df = pd.DataFrame(validation_data, columns=['text', 'label'])
x_valid = valid_df['text'].tolist()
y_valid = valid_df['label'].apply(lambda x: x['cats']['AGE_VERIFICATION']).tolist()


test_data = [ # This dataset should be used whenever a grid search is performed, in order to avoid optimisation of the validation set thus resulting in better generality
    ("Are you over age?", {"cats": {"AGE_VERIFICATION": 1.0}}),
    
]
random.shuffle(test_data)
test_df = pd.DataFrame(test_data, columns=['text', 'label'])
x_test = test_df['text'].tolist()
y_test = test_df['label'].apply(lambda x: x['cats']['AGE_VERIFICATION']).tolist()

# Export data for use in other modules
def get_data():
    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)