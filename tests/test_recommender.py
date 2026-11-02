import pandas as pd
import pytest
from src.recommender import ContentRecommender, build_model, get_recommendations

def sample_data():
    movies = pd.DataFrame({"movieId":[1,2,3,4],"title":["A","B","C","D"],
                           "genres":["Action|Comedy","Action","Comedy","Drama"]})
    tags = pd.DataFrame({"movieId":[1,2,3],"tag":["funny hero","hero action","funny"]})
    ratings = pd.DataFrame({"movieId":[1,2,2,3],"rating":[4,4,5,3]})
    return movies,tags,ratings

def test_recommendations_exclude_source_and_return_expected_count():
    movies,tags,ratings=sample_data(); model=ContentRecommender(movies,tags,ratings)
    result=model.recommend("A",top_n=2)
    assert len(result)==2
    assert "A" not in result.title.tolist()
    assert result.similarity.between(0,1).all()
    assert result.iloc[0].explanation

def test_unknown_title_has_clear_error():
    movies,tags,ratings=sample_data(); model=ContentRecommender(movies,tags,ratings)
    with pytest.raises(ValueError,match="tidak ditemukan"):
        model.recommend("judul tidak ada")

def test_top_n_validation():
    movies,tags,ratings=sample_data(); model=ContentRecommender(movies,tags,ratings)
    with pytest.raises(ValueError): model.recommend("A",top_n=0)

def test_legacy_api_works():
    movies,_,_=sample_data(); sim,df=build_model(movies)
    result=get_recommendations("A",df,sim,top_n=2)
    assert len(result)==2
