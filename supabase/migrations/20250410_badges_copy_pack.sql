alter table badges
    alter column rarity set default 'common',
    alter column lore set default '',
    alter column hint set default '',
    alter column scope set default 'overall';

update badges
set rarity = coalesce(rarity, 'common')
where rarity is null;
