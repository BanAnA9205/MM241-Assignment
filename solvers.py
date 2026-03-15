import numpy as np
from core_policy import Policy

class Greedy(Policy):
    def __init__(self):
        super().__init__()
        self.actions = []

    def get_action(self, observation, info):
        if info["filled_ratio"] == 0:
            stocks = np.array(observation["stocks"]) == -1
            s_size = np.vstack((np.sum(stocks[:, :, 0], axis=1), np.sum(stocks[:, 0], axis=1))).T

            # Rotate stocks if necessary so width > height
            s_rotate = s_size[:, 0] > s_size[:, 1]
            s_size[s_rotate] = s_size[s_rotate, ::-1]
            stocks[s_rotate] = np.transpose(stocks[s_rotate], (0, 2, 1))

            p_size = np.array([product["size"] for product in observation["products"]])
            p_size.sort(axis=1)
            
            # Combine sizes and quantities, sort by height descending
            quantities = np.array([prod["quantity"] for prod in observation["products"]])
            products = np.column_stack((p_size, quantities))[np.argsort(-p_size[:, 1])]

            self.actions = []

            # Try to place products into stocks sorted by descending height
            for i in np.argsort(-s_size[:, 1]):
                # Pass 1: standard orientation
                for product in products:
                    if product[2] == 0: continue
                    x_range = range(s_size[i, 1] - product[1] + 1)
                    y_range = range(s_size[i, 0] - product[0] + 1)
                    
                    for y in y_range:
                        for x in x_range:
                            if np.all(stocks[i, y:y + product[0], x:x + product[1]]):
                                stocks[i, y:y + product[0], x:x + product[1]].fill(False)
                                product[2] -= 1
                                size = product[[1, 0]] if s_rotate[i] else product[[0, 1]]
                                pos = (x, y) if s_rotate[i] else (y, x)
                                self.actions.append({"stock_idx": i, "size": size, "position": pos})
                                if product[2] == 0: break
                        if product[2] == 0: break
                
                products = products[products[:, 2] > 0]

                # Pass 2: rotated orientation
                for product in products:
                    if product[2] == 0: continue
                    x_range = range(s_size[i, 1] - product[0] + 1)
                    y_range = range(s_size[i, 0] - product[1] + 1)
                    
                    for y in y_range:
                        for x in x_range:
                            if np.all(stocks[i, y:y + product[1], x:x + product[0]]):
                                stocks[i, y:y + product[1], x:x + product[0]].fill(False)
                                product[2] -= 1
                                size = product[[0, 1]] if s_rotate[i] else product[[1, 0]]
                                pos = (x, y) if s_rotate[i] else (y, x)
                                self.actions.append({"stock_idx": i, "size": size, "position": pos})
                                if product[2] == 0: break
                        if product[2] == 0: break
                
                products = products[products[:, 2] > 0]
                if len(products) == 0: break

        return self.actions.pop()

class Genetic(Policy):
    def __init__(self):
        super().__init__()
        self.best_chromosome = None     
        self.best_cut = None            
        self.best_score = 1             
        self.idx = 0
        self.population_size = 100

    # A. Encoding
    def encode(self):
        # Pythonic way to generate codes instead of manual loops
        self.sheetsCode = np.arange(len(self.sheets))
        
        # Repeat piece indices based on their required quantities
        quantities = [p[2] for p in self.pieces]
        self.piecesCode = np.repeat(np.arange(len(self.pieces)), quantities)
        
        self.sheets_length = len(self.sheets)
        self.pieces_length = len(self.piecesCode)
        self.chromosomes_length = self.sheets_length + self.pieces_length

    # B. Initial Population
    def init_population(self):
        # List comprehension directly translates to the required population matrix
        self.chromosomes = np.array([
            np.concatenate((np.random.permutation(self.sheetsCode), 
                            np.random.permutation(self.pieces_length)))
            for _ in range(self.population_size)
        ])

    def closer_to_bottom_left(self, a, b, c, d):
        dist1, dist2 = a*a + b*b, c*c + d*d
        return dist1 < dist2 or (dist1 == dist2 and a < c)

    def intersection(self, r1, r2):
        x1, y1, x2, y2 = r1
        x3, y3, x4, y4 = r2
        if x1 >= x4 or x2 <= x3 or y1 >= y4 or y2 <= y3:
            return None
        return max(x1, x3), max(y1, y3), min(x2, x4), min(y2, y4)

    def dif_elim(self, inter, ers):
        x1, y1, x2, y2 = ers
        ix1, iy1, ix2, iy2 = inter
        new_ers = []
        
        if x1 < ix1: new_ers.append((x1, y1, ix1, y2))
        if ix2 < x2: new_ers.append((ix2, y1, x2, y2))
        if y1 < iy1: new_ers.append((x1, y1, x2, iy1))
        if iy2 < y2: new_ers.append((x1, iy2, x2, y2))
        
        return new_ers if new_ers else None

    # C & D. Guillotine Cut Process and Heuristic Placement
    def guillotine_cut(self, chromosome):
        # Use native Python lists instead of np.empty(..., dtype=object) for better performance
        ERS = [[(0, 0, self.sheets[chromosome[i]][0], self.sheets[chromosome[i]][1])] 
               for i in range(self.sheets_length)]
        
        largest_ERS = np.zeros(self.sheets_length, dtype=int)
        small_piece = np.full(self.sheets_length, np.inf)
        placed = [None] * self.chromosomes_length
        
        for i in range(self.sheets_length):
            for j in range(self.sheets_length, self.chromosomes_length):
                if placed[j] is not None: continue
                
                Pj = self.pieces[self.piecesCode[chromosome[j]]]
                ERSk_x, ERSk_y = np.inf, np.inf
                
                for ers in ERS[i]:
                    if ers[0] + Pj[0] <= ers[2] and ers[1] + Pj[1] <= ers[3]:
                        if self.closer_to_bottom_left(ers[0], ers[1], ERSk_x, ERSk_y):
                            ERSk_x, ERSk_y = ers[0], ers[1]
                            
                if ERSk_x == np.inf: continue
                
                placed[j] = (chromosome[i], ERSk_x, ERSk_y, ERSk_x + Pj[0], ERSk_y + Pj[1])
                small_piece[i] = min(small_piece[i], Pj[0] * Pj[1])

                new_ers = []
                to_remove = set()
                
                for ide, ers in enumerate(ERS[i]):
                    inter = self.intersection((ERSk_x, ERSk_y, ERSk_x + Pj[0], ERSk_y + Pj[1]), ers)
                    if inter:
                        to_remove.add(ide)
                        temp_ers = self.dif_elim(inter, ers)
                        if temp_ers: new_ers.extend(temp_ers)

                # Cross-check new ERSs
                valid_new_ers = []
                for n_ers in new_ers:
                    # Check if eclipsed by surviving old ERS
                    if any(ers[0] <= n_ers[0] and ers[1] <= n_ers[1] and n_ers[2] <= ers[2] and n_ers[3] <= ers[3] 
                           for ide, ers in enumerate(ERS[i]) if ide not in to_remove):
                        continue
                    # Check if eclipsed by other new ERS
                    if any(n2[0] <= n_ers[0] and n2[1] <= n_ers[1] and n_ers[2] <= n2[2] and n_ers[3] <= n2[3] 
                           for n2 in new_ers if n2 is not n_ers):
                        continue
                    valid_new_ers.append(n_ers)

                ERS[i] = [ers for ide, ers in enumerate(ERS[i]) if ide not in to_remove] + valid_new_ers

            # Calculate largest ERS area for this sheet
            if ERS[i]:
                largest_ERS[i] = max((e[2] - e[0]) * (e[3] - e[1]) for e in ERS[i])

        return placed[self.sheets_length:], largest_ERS, small_piece

    def score(self, chromosome):
        placed, largest_ERS, small_piece = self.guillotine_cut(chromosome)
        score, minus, sheet_area = 0, 0, 0
        sheet_mark = np.zeros(self.sheets_length, dtype=bool)

        for p in placed:
            if p is not None:
                score += (p[3] - p[1]) * (p[4] - p[2])
                if not sheet_mark[p[0]]:
                    sheet_area += self.sheets[p[0]][0] * self.sheets[p[0]][1]
                    sheet_mark[p[0]] = True

        for i in range(self.sheets_length):
            if small_piece[i] != np.inf:
                minus += (small_piece[i] / sheet_area) * (largest_ERS[i] / sheet_area)

        return 1 - score / sheet_area - 0.02 * minus

    def breed(self, parent1, parent2):
        length = len(parent1)
        newchild = np.full(length, -1, dtype=int)
        chosen = np.zeros(length, dtype=bool)
        
        mask = np.random.rand(length) < 0.5
        
        for i in range(length):
            g1, g2 = parent1[i], parent2[i]
            if mask[i]:
                if not chosen[g1]:
                    newchild[i], chosen[g1] = g1, True
                elif not chosen[g2]:
                    newchild[i], chosen[g2] = g2, True
            else:
                if not chosen[g2]:
                    newchild[i], chosen[g2] = g2, True
                elif not chosen[g1]:
                    newchild[i], chosen[g1] = g1, True

        # Fill any remaining -1 spots with unchosen genes
        missing = [g for g in parent1 if not chosen[g]]
        newchild[newchild == -1] = missing
        
        return newchild

    # E, F & G: Reproduction, Crossover, Mutation
    def next_generation(self):
        scores = np.array([self.score(c) for c in self.chromosomes])
        
        # Sort by score ascending (lower score is better based on objective func)
        sorted_idx = np.argsort(scores, kind='mergesort')
        self.chromosomes = self.chromosomes[sorted_idx]
        
        do_mutation = np.random.rand() < 0.1
        reproduct_range = int(0.2 * self.population_size)
        
        new_generation = np.empty_like(self.chromosomes)
        new_generation[:reproduct_range] = self.chromosomes[:reproduct_range]
        
        limit = int(0.8 * self.population_size) if do_mutation else self.population_size
        
        for index in range(reproduct_range, limit):
            parent1 = new_generation[np.random.randint(reproduct_range)]
            parent2 = self.chromosomes[np.random.randint(self.population_size)]
            
            new_sheets = self.breed(parent1[:self.sheets_length], parent2[:self.sheets_length])
            new_pieces = self.breed(parent1[self.sheets_length:], parent2[self.sheets_length:])
            new_generation[index] = np.concatenate((new_sheets, new_pieces))
            
        if do_mutation:
            for index in range(limit, self.population_size):
                sheets = np.random.permutation(self.sheetsCode)
                pieces = np.random.permutation(self.pieces_length)
                new_generation[index] = np.concatenate((sheets, pieces))
                
        self.chromosomes = new_generation

    def iteration(self, num_iter):
        best_chromosome, best_cut, best_score = None, None, 1
        for _ in range(num_iter):
            self.next_generation()
            current_best = self.chromosomes[0]
            current_score = self.score(current_best)
            
            if current_score < best_score:
                best_chromosome = current_best
                best_score = current_score
                
        # Only evaluate the guillotine cut for the very best found to save time
        best_cut, _, _ = self.guillotine_cut(best_chromosome)
        return best_chromosome, best_cut, best_score

    def mpga(self, num_phase=8, num_iter=20):
        self.encode()
        self.init_population()
        phases_not_improved = 0
        
        for _ in range(num_phase):
            best_chromosome, best_cut, best_score = self.iteration(num_iter)
            
            if best_score < self.best_score:
                self.best_chromosome = best_chromosome
                self.best_cut = best_cut
                
                if self.best_score - best_score < 1e-4:
                    phases_not_improved += 1
                else:
                    phases_not_improved = 0
                self.best_score = best_score
            
            if phases_not_improved == 2:
                break
                
            # Retain top 15% and re-initialize the rest
            top_15_count = int(0.15 * self.population_size)
            top_chromosomes = self.chromosomes[:top_15_count].copy()
            self.init_population()
            self.chromosomes[:top_15_count] = top_chromosomes

    def get_action(self, observation, info):
        if self.best_chromosome is None:
            self.sheets = []
            self.pieces = []
            self.sheetsArea = 0
            self.sheet_mark = np.zeros(len(observation["stocks"]), dtype=bool)

            for stock in observation["stocks"]:
                stock_w, stock_h = self._get_stock_size_(stock)
                self.sheetsArea += stock_w * stock_h
                self.sheets.append([stock_w, stock_h])
                
            for prod in observation["products"]:
                self.pieces.append([prod["size"][0], prod["size"][1], prod["quantity"]])

            self.mpga()
        
        while self.idx < len(self.best_cut):
            placement_info = self.best_cut[self.idx]
            self.idx += 1
            if placement_info is None:
                continue
            
            stock_idx = placement_info[0]
            size = (placement_info[3] - placement_info[1], placement_info[4] - placement_info[2])
            position = (placement_info[1], placement_info[2])
            
            return {"stock_idx": stock_idx, "size": size, "position": position}
        
        # Reset for next episode
        self.best_chromosome = None
        self.best_cut = None
        self.best_score = 1
        self.idx = 0
        return self.get_action(observation, info)