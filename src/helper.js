const url = "https://pokeapi.co/api/v2/pokemon"

for(let i = 1; i < 152; i++) {
    const res = await fetch(`${url}/${i}`)

    const name = res.species.name;
    const image_url = res.sprites.front_default
}

